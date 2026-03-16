import cv2
import os

net = cv2.dnn.readNetFromCaffe('data/MobileNet/mobilenet_deploy.prototxt', 'data/MobileNet/mobilenet.caffemodel')

classes = []

with open('data/MobileNet/synset.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        
        if not line:
            continue
        
        parts = line.split(' ', 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)

path_to_folder = 'images/MobileNet'

info = {}

if not os.path.exists(path_to_folder):
    raise FileNotFoundError
else:
    for filename in os.listdir(path_to_folder):
        if filename.lower().endswith(('.png', '.jpg', '.webp')):
            img_path = os.path.join(path_to_folder, filename)
            image = cv2.imread(img_path)
            
            if image is None:
                continue

            blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224)), 1.0 / 127.5, (224, 224), (127.5, 127.5, 127.5), swapRB=True)
            net.setInput(blob)
            preds = net.forward()

            index = preds[0].argmax()
            label = classes[index] if index < len(classes) else 'Undefined'
            conf = float(preds[0][index].item()) * 100

            print(f'Файл: {filename}\nКлас: {label} ({conf:.2f}%)')

            info[label] = info.get(label, 0) + 1

            text = f"{label}: {conf:.2f}%"
            cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('MobileNet', image)
            
            cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('\n' + '='*40)
    print('Клас | Кількість')
    print("-"*40)
    
    for calcs, count in info.items():
        print(f"{calcs} | {count}")
        
    print("="*40)