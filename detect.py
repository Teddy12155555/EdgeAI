import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import os


workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TH = 1

print('Running on device: {}'.format(device))

def collate_fn(x):
    return x[0]

def detect(save_img=False, detector=None):
    source, weights,imgsz = 'input', 'weights.pt',640

    # Initialize
    #set_logging()
    device = select_device('0')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic='NMS')
        
        transform1 = transforms.Compose([
	        transforms.ToTensor(),
            transforms.Resize(160) 
	        ]
        )
        # Process detections
        for _, det in enumerate(pred):  # detections per image
            _, s, im0, _ = Path(path), '', im0s, getattr(dataset, 'frame', 0)

            s += '%gx%g ' % img.shape[2:]  # print string
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}s, '  # add to string

                # Write results
                #faces = detector.mtcnn_detect(im0)
                SPAND = 25
                h, w, c = im0.shape
                name = 'unknown'
                for *xyxy, conf, cls in reversed(det):
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    y_u, y_d, x_l, x_r = 0,h,0,w
                    if c1[1] - SPAND > 0:
                        y_u = c1[1] - SPAND
                    if c2[1] + SPAND < h:
                        y_d = c2[1] + SPAND
                    if c1[0] - SPAND > 0:
                        x_l = c1[0] - SPAND
                    if c2[0] + SPAND < w:
                        x_r  = c2[0] + SPAND
                    onestage = im0[y_u:y_d, x_l:x_r]

                    ten = transform1(onestage)
                    face_embedding = detector.resnet(ten.unsqueeze(0).to(device))
                    probs = [(face_embedding - detector.embeddings[i]).norm().item() for i in range(detector.embeddings.size()[0])] 
                    index = probs.index(min(probs))
                    name = detector.names[index]
                    print('')
                    print(name)
                    
                    label = f'{names[int(cls)]} {name}'

                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

            t2 = time_synchronized()
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            cv2.imwrite('Result.jpg', im0)

class FaceNet():
    def __init__(self):
        super(FaceNet)
        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=device
        )
        self.resnet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)
        #self.resnet = InceptionResnetV1()
        #checkpoint = torch.load('save.pt')
        #self.resnet.load_state_dict(checkpoint['state_dict'])
        
        #self.resnet = self.resnet.eval().to(device)
        
        self.dataloader = None
        self.aligned = None
        self.names = None
        self.embeddings = None
        
    def gen_db(self,root, init=True):
        if init:
            dataset = datasets.ImageFolder(f'{root}/raw')
            dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
            self.dataloader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers) 

            aligned = []
            names = []
            i = 1
            if not os.path.exists(f'{root}/aligned/'):
                os.mkdir(f'{root}/aligned/')

            classname = ''
            for x, y in self.dataloader:
                path = f'{root}/aligned/{dataset.idx_to_class[y]}/'
                if classname != dataset.idx_to_class[y]:
                    classname = dataset.idx_to_class[y]
                    print(f'Create Face => {classname}')
                if not os.path.exists(path):
                    i = 1
                    os.mkdir(path)
                x_aligned, prob = self.mtcnn(x, return_prob=True,save_path= f'{path}/{i}.jpg')
                i += 1
                if x_aligned is not None:
                    aligned.append(x_aligned) 
                    names.append(dataset.idx_to_class[y])   
        
            self.aligned = aligned
            self.names = names

            self.save_db()
        else:
            dataset = datasets.ImageFolder(f'{root}/new')
            dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
            loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)
            i = 1
            if not os.path.exists(f'{root}/aligned/'):
                os.mkdir(f'{root}/aligned/')

            classname = ''
            for x, y in loader:
                path = f'{root}/aligned/{dataset.idx_to_class[y]}/'
                if classname != dataset.idx_to_class[y]:
                    classname = dataset.idx_to_class[y]
                    print(f'Add New Face => {classname}')
                if not os.path.exists(path):
                    i = 1
                    os.mkdir(path)
                x_aligned, prob = self.mtcnn(x, return_prob=True,save_path= f'{path}/{i}.jpg')
                i += 1
                if x_aligned is not None:
                    self.aligned.append(x_aligned) 
                    self.names.append(dataset.idx_to_class[y])

            self.save_db()

    def save_db(self):
        al_tensor = torch.stack(self.aligned).to(device)
        self.embeddings = self.resnet(al_tensor).detach().cpu()
        torch.save(self.embeddings,'embeddings.pt')
        torch.save(self.names,'names.pt')

class FaceNetDetector():
    def __init__(self, net):
        super(FaceNetDetector)
        # detect
        self.mtcnn_detect = MTCNN(keep_all=True, device=device)
        self.resnet = net
        self.update()
    
    def update(self):
        self.names = torch.load("./names.pt")
        self.embeddings = torch.load("./embeddings.pt").to(device)

    def detect_frame(self, img, save=False):
        # @input -> img = PIL Image
        # @return -> cvimage = CV2 Image
        cvimage = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        faces = self.mtcnn_detect(img)
        boxes, _ = self.mtcnn_detect.detect(img) 
        
        if boxes is None:
            return cvimage

        for i,box in enumerate(boxes):
            [x0,y0,x1,y1] = box.tolist()
            cv2.rectangle(cvimage, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 3, cv2.LINE_AA)
            face_embedding = self.resnet(faces[i].unsqueeze(0).to(device))
        
            probs = [(face_embedding - self.embeddings[i]).norm().item() for i in range(self.embeddings.size()[0])] 
            index = probs.index(min(probs))
            #print(min(probs))
            if min(probs) > TH:
                name = 'unknown' 
            else:
                name = self.names[index]
            
            cv2.putText(cvimage, str(name), (int(box[0]) + 100,int(box[1]) + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 5, cv2.LINE_AA)
            if save:
                cv2.imwrite('cv_result.jpg',cvimage)
        return cvimage

if __name__ == '__main__':
    net = FaceNet()
    # init
    net.gen_db('./face_db',True)
    # add new one
    net.gen_db('./face_db',False)

    detector = FaceNetDetector(net.resnet)
    with torch.no_grad():
        detect(save_img=True,detector=detector)
        
