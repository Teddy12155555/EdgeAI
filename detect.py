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
    source, weights, view_img, imgsz = opt.source, opt.weights, opt.view_img,  opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    
    # Initialize
    #set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
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
        print('!!!')
        print(len(pred))
        for i, det in enumerate(pred):  # detections per image
            if webcam: 
                p, s, im0, frame = Path(path[i]), '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = Path(path), '', im0s, getattr(dataset, 'frame', 0)

            save_path = str(save_dir / p.name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}s, '  # add to string

                # Write results
                #faces = detector.mtcnn_detect(im0)
                xyxy_list = []
                label_list = []
                DEBUG = 1
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
                    #cv2.rectangle(im0, (x_l, y_u), (x_r, y_d), (0,0,255), 3, cv2.LINE_AA)
                    ten = transform1(onestage)
                    face_embedding = detector.resnet(ten.unsqueeze(0).to(device))
                    probs = [(face_embedding - detector.embeddings[i]).norm().item() for i in range(detector.embeddings.size()[0])] 
                    index = probs.index(min(probs))
                    print(min(probs))
                    #print(max(probs))
                    name = detector.names[index]
                    print(name)
                    if (save_img or view_img) and 1:  # Add bbox to image
                        #faces = detector.mtcnn_detect(onestage)
                        # if (faces is not None) and DEBUG:
                        #     cv2.rectangle(im0, (x_l, y_u), (x_r, y_d), (0,0,255), 3, cv2.LINE_AA)
                        #boxes, _ = detector.mtcnn_detect.detect(onestage)
                        
                        #cv2.imshow('TEST', onestage)
                        #cv2.waitKey(0)
                        """
                        if (boxes is not None) and (faces is not None):
                            for i,box in enumerate(boxes):
                                face_embedding = detector.resnet(faces[i].unsqueeze(0).to(device))
                                probs = [(face_embedding - detector.embeddings[i]).norm().item() for i in range(detector.embeddings.size()[0])] 
                                index = probs.index(min(probs))
                                print(min(probs))
                                if min(probs) > TH:
                                    name = 'unknown' 
                                else:
                                    name = detector.names[index]
                        
                        """
                        #label = f'{names[int(cls)]} {conf:.2f}'
                        label = f'{names[int(cls)]} {name}'

                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)   
                    #print(label_list)
                    #print(xyxy_list)
            # Print time (inference + NMS)
                #cv2.imshow('DEBUG', im0)
                #cv2.waitKey(0)
            t2 = time_synchronized()
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite('result.jpg', im0)
                    print(f'Save! in {save_path}')
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter('result.mp4', cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    #print(f'Done. ({time.time() - t0:.3f}s)')

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
            for x, y in self.dataloader:
                path = f'{root}/aligned/{dataset.idx_to_class[y]}/'
                if not os.path.exists(path):
                    i = 1
                    os.mkdir(path)
                x_aligned, prob = self.mtcnn(x, return_prob=True,save_path= f'{path}/{i}.jpg')
                i += 1
                if x_aligned is not None:
                    print('Face detected with probability: {:8f}'.format(prob))
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
            for x, y in loader:
                path = f'{root}/aligned/{dataset.idx_to_class[y]}/'
                if not os.path.exists(path):
                    i = 1
                    os.mkdir(path)
                x_aligned, prob = self.mtcnn(x, return_prob=True,save_path= f'{path}/{i}.jpg')
                i += 1
                if x_aligned is not None:
                    print('Face detected with probability: {:8f}'.format(prob))
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    
    net = FaceNet()
    # init
    #net.gen_db('./face_db',True)
    # add new one
    #net.gen_db('./face_db',False)

    detector = FaceNetDetector(net.resnet)
    with torch.no_grad():
        detect(save_img=True,detector=detector)
        
