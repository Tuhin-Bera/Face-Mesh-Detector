import cv2
import mediapipe as mp
import time





class face_mesh_detector():
    def __init__(self, static_mode = False, max_faces = 2, min_detection_con = 0.5, min_track_con = 0.5):
        self.static_mode = static_mode
        self.max_faces = max_faces
        self.min_detection_con = min_detection_con
        self.min_track_con = min_track_con

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=self.static_mode, 
            max_num_faces=self.max_faces, 
            min_detection_confidence=self.min_detection_con, 
            min_tracking_confidence=self.min_track_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.draw_spec = self.mp_draw.DrawingSpec(thickness = 1, circle_radius = 1)

            

    def find_face_mesh(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(imgRGB)
        
        faces = []
        
        if self.results.multi_face_landmarks:
            for face_lms in self.results.multi_face_landmarks:
                if draw:
                    # self.mp_draw.draw_landmarks(img, face_lms, self.mp_face_mesh.FACE_CONNECTIONS)           ## it is for older version, it will not work anymore in this code
                    # self.mp_draw.draw_landmarks(img, face_lms, self.mp_face_mesh.FACEMESH_TESSELATION)        ## these are the alternate replacement of "FACE_CONNECTIONS"
                    self.mp_draw.draw_landmarks(img, face_lms, self.mp_face_mesh.FACEMESH_CONTOURS, self.draw_spec, self.draw_spec)
                    # self.mp_draw.draw_landmarks(img, face_lms, self.mp_face_mesh.FACEMESH_IRISES)               # For eye landmarks
                    face = []
                    for id, lm in enumerate(face_lms.landmark):
                        # print(lm)
                        ih, iw, ic = img.shape
                        x, y = int(lm.x * iw), int(lm.y * ih)
                        
                        # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)       ## this line is not suitable for this code, but it can be useful sometimes
                        
                        # print(id, x, y)
                        face.append([x, y])
                    faces.append(face)
                
        return img, faces

    
    
    
    
def main():
    # cap = cv2.VideoCapture('V:/part 10/face2.mp4')      # it is use for a video
    cap = cv2.VideoCapture(0)                         # we can use it for the camera also

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
    
    time.sleep(2)

    p_time = 0
    c_time = 0
    
    detector = face_mesh_detector()
    
    while True:
        success, img = cap.read()
        
        if not success or img is None:
            print("Error: Failed to capture image")  
            break  # Exit if no frame is captured
        
        
        img, faces = detector.find_face_mesh(img)
        
        if len(faces) != 0:
            print(faces[0])
            
            
        c_time  = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        
        cv2.putText(img, f'FPS: {str(int(fps))}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        
        cv2.imshow("Image", img)   # display the proccessed fame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Press 'q' to exit
    
    
    
    
    
    
    
    
if __name__=="__main__":
    main()