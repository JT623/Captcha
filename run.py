import os
import torch
import torch.nn as nn
from Testmodel import CNN
from datasets import CaptchaData
from torchvision.transforms import Compose, ToTensor
import pandas as pd

model_path = './checkpoints/model.pth'
testpath = './data/test/'
result_path = '/result/submission.csv'

source = [str(i) for i in range(0, 10)]
source += [chr(i) for i in range(97, 97+26)]
source += [chr(i) for  i in range(65,65+26)]
alphabet = ''.join(source)

def predict(img_dir=testpath):
    transforms = Compose([ToTensor()])
    dataset = CaptchaData(img_dir, transform=transforms)
   # dataset = MyDataSet()
    cnn = CNN()
    if torch.cuda.is_available():
        cnn = cnn.cuda()
        cnn.eval()
        cnn.load_state_dict(torch.load(model_path))
    else:
        cnn.eval()
        model = torch.load(model_path, map_location='cpu')
        cnn.load_state_dict(model)

    for k, (img, target) in enumerate(dataset):
        #img = img.view(1, 3, 40, 120).cuda()
        #target = target.view(1, 4*62).cuda()
        #target = target.view(1, 4*62).cuda()
        img = img.view(1, 3, 40, 120)
        output = cnn(img)

        output = output.view(-1, 62)
        target = target.view(-1, 62)
        output = nn.functional.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        output = output.view(-1, 4)[0]
        #target = target.view(-1, 4)[0]
        a=''.join([alphabet[i] for i in output.cpu().numpy()])
        b =''.join([alphabet[i] for i in target.cpu().numpy()])+'.jpg'


        #print('ID: ' + ''.join([alphabet[i] for i in target.cpu().numpy()]) + '.jpg')
        #print('label: '+''.join([alphabet[i] for i in output.cpu().numpy()]))

        dataframe = pd.DataFrame({'ID':b,'label':a},index=[0])
        dataframe.to_csv('1.csv',index=False,mode='a')

    with open('1.csv','r') as r:
        lines = r.readlines()
    with open('2.csv','w') as w:
        for i in lines:
            if 'label' not in i:
                w.write(i)

    df= pd.read_csv('2.csv',header=None,names=['ID','label'])
    df.to_csv('3.csv',index=False)

    with open('3.csv','r') as r:
        lines = r.readlines()
    with open(result_path,'w') as w:
        for l in lines:
           str = l.strip('0')
           w.write(str)

    f = pd.read_csv(result_path)
    print(f)

    os.remove('1.csv')
    os.remove('2.csv')
    os.remove('3.csv')

if __name__=="__main__":
    predict()