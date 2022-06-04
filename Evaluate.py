import matplotlib.pyplot as plt
import matplotlib
from CustomDataLoader import DogCats
import torch
from torch.autograd import Variable
matplotlib.use('TKAgg')


dataset_path = './validation_data_cutre/cats/*'
dl_cats = DogCats((224, 224), dataset_path)
dataset_path = './validation_data_cutre/dogs/*'
dl_dogs = DogCats((224, 224), dataset_path)

CLASSES = ['dog', 'cat']

model_path = f'./checkpoints/vit_model_32.pth'
model = torch.load(model_path)
model = model.cuda()

result = []
for dog in dl_dogs:
    image, original = dog['image'], dog['original']
    image = torch.unsqueeze(image, dim=0)
    image = Variable(image).cuda()

    output_tensor = model(image)
    output_probs = torch.exp(output_tensor)
    class_prediction = output_probs.data.max(1, keepdim=True)[1][0][0].cpu().data.numpy()

    # plt.title(CLASSES[class_prediction])
    # plt.imshow(original)
    # plt.show()

    result.append(class_prediction == 0)

for cat in dl_cats:
    image, original = cat['image'], cat['original']
    image = torch.unsqueeze(image, dim=0)
    image = Variable(image).cuda()

    output_tensor = model(image)
    output_probs = torch.exp(output_tensor)
    class_prediction = output_probs.data.max(1, keepdim=True)[1][0][0].cpu().data.numpy()

    # plt.title(CLASSES[class_prediction])
    # plt.imshow(original)
    # plt.show()

    result.append(class_prediction == 1)

print(sum(result) / len(result) * 100.)
