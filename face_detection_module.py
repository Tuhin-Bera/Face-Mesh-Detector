import cv2
import mediapipe as mp
import time 


class face_detector():
    def __init__(self, min_detection_con = 0.5):
        self.min_detection_con = min_detection_con

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(self.min_detection_con)
        self.mp_draw = mp.solutions.drawing_utils
        
        
      
        
    def find_faces(self, img, draw = True):    
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(img_RGB)
        print(self.results)
        b_boxes = []
        
    
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                b_box_c = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                b_box = int(b_box_c.xmin*iw), int(b_box_c.ymin*ih), \
                    int(b_box_c.width*iw), int(b_box_c.height*ih)
                    
                b_boxes.append([id, b_box, detection.score])
                
                if draw:
                    img = self.fancy_draw(img, b_box)
                    
                    cv2.putText(img,f'{int(detection.score[0] * 100)}%', (b_box[0], b_box[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)    ## placing the score in the video
                
        return img, b_boxes
    
    
    
    def fancy_draw(self, img, b_box, l = 30,t = 5, rt = 1):
        x, y, w, h = b_box
        x1, y1 = x + w, y + h
        cv2.rectangle(img, b_box, (255, 0, 0), rt)                        ## we are drawing rectangle(cv2) without using mediapipe draw_detection
        ## top left corner (x, y)
        cv2.line(img, (x, y), (x + l, y), (255, 0, 0), t)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 0), t)
        
        ## top right corner (x1, y)
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 0), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 0), t)
        
        ## top left corner (x, y1)
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 0), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 0), t)
        
        ## top left corner (x1, y1)
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 0), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 0), t)
        
        
        return img
    
    

def main():
    cap = cv2.VideoCapture('V:/part 10/face.mp4')        ## capturing faces in the video
    # cap = cv2.VideoCapture(0)                              ## capturing faces through the camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
    
    time.sleep(2)

    p_time = 0
    c_time = 0
    
    detector = face_detector()
    
    while True:
        success, img = cap.read()
    
        if not success or img is None:
            print("Error: Failed to capture image")  
            break  # Exit if no frame is captured
    
        img, b_boxes =detector.find_faces(img)
        print(b_boxes)
        
        
        c_time  = time.time()
        fps = 1/(c_time - p_time)
        p_time = c_time
        
        cv2.putText(img,f'FPS: {str(int(fps))}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        cv2.imshow("Image", img)   # display the proccessed fame

        if cv2.waitKey(1) & 0xFF == ord('q'):               ## by increasing the value of waitkey, we can decrease the fps value so it will be slower
            break  # Press 'q' to exit
    
    
    


if __name__ == "__main__":
    main()