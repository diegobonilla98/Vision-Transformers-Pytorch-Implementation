import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import repeat, rearrange
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TKAgg')


class TransformerBlock(nn.Module):
    def __init__(self, D, H):
        super(TransformerBlock, self).__init__()

        self.D = D
        self.HEADS = H

        self.to_qkv = nn.Linear(self.D, self.D * 3, bias=False)

        self.ln = nn.LayerNorm(self.D)
        self.mh_att = nn.MultiheadAttention(self.D, self.HEADS, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(self.D, self.D * 4),
            nn.GELU(),
            nn.Linear(self.D * 4, self.D),
        )

    def forward(self, x0):
        x = self.ln(x0)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        x, _ = self.mh_att(q, k, v)
        x += x0
        x0 = x
        x = self.ln(x)
        x = self.mlp(x)
        x += x0
        return x


class ViT(nn.Module):
    def __init__(self, num_classes, embedding_size=768, num_layers=12, num_patches=16, num_heads=12, image_size=224):
        super(ViT, self).__init__()

        self.W = self.H = image_size
        self.HEADS = num_heads
        self.C = 3
        self.N = num_patches
        self.P = int(self.W / np.sqrt(self.N))
        self.D = embedding_size
        self.L = num_layers
        self.NUM_CLASSES = num_classes

        self.pos_enc = []
        for n in range(self.N + 1):
            n_pos_enc = []
            for d in range(self.D):
                if d % 2 == 0:
                    val = np.sin(n / 10_000 ** (2 * d / self.D))
                else:
                    val = np.cos(n / 10_000 ** (2 * d / self.D))
                n_pos_enc.append(val)
            self.pos_enc.append(n_pos_enc)
        self.pos_enc = np.array(self.pos_enc, np.float32)
        self.pos_enc = torch.from_numpy(self.pos_enc)
        self.pos_enc = self.pos_enc.cuda()

        self.patchify = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.P, p2=self.P)
        self.patch_embedding = nn.Linear(self.P * self.P * self.C, self.D)

        self.transformer_encoder = nn.Sequential(
            *[
                TransformerBlock(self.D, self.HEADS) for _ in range(self.L)
            ]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.D),  # ADDED (https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py)
            nn.Linear(self.D, self.NUM_CLASSES),
            nn.LogSoftmax()
        )

    def forward(self, x):
        x = self.patchify(x)  # R(B, H, W, C) -> R(B, N, (P²*C))
        x = self.patch_embedding(x)  # R(B, N, (P²*C)) -> R(B, N, D)

        x_shape = x.shape
        class_embedding = torch.randn((x_shape[0], 1, self.D)).cuda()  # CHANGED FROM RANDOM TO ZEROS (https://github.com/google-research/vision_transformer/issues/61)
        x = torch.cat((class_embedding, x), 1)  # R(B, N, D) -> R(B, N+1, D)

        pos_enc = repeat(self.pos_enc, 'n d -> b n d', b=x_shape[0])
        x += pos_enc

        x = self.transformer_encoder(x)  # R(B, N + 1, D) -> R(B, N+1, D)

        x = x[:, 0, :]  # R(B, N, D) -> R(B, D)
        x = self.classifier(x)  # R(B, D) -> R(B, N_CLS)

        return x


if __name__ == '__main__':
    from torch.autograd import Variable

    model = ViT(10)
    model = model.cuda()

    # image = cv2.imread('test.jpg')[:, :, ::-1]  # BGR -> RGB
    # image = cv2.resize(image, (64, 64))
    # image = np.expand_dims(image, axis=0).astype(np.float32)
    # input_tensor = torch.from_numpy(image.copy())
    # input_tensor = input_tensor.permute(0, 3, 1, 2)

    input_tensor = torch.randn((1, 3, 224, 224))

    input_tensor = input_tensor.cuda()
    input_tensor = Variable(input_tensor)

    output_tensor = model(input_tensor)

    print()
