import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from settings import COLORS

class GraspingGUI:
    def __init__(self, root, vision_system, robot_arm):
        self.root = root
        self.vision = vision_system
        self.arm = robot_arm
        
        self.setup_ui()
        self.update_video()
    
    def setup_ui(self):
        self.root.title("Robot Arm Grasping System")
        self.root.configure(bg=COLORS['background'])
        
        # Main frames
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Video frame
        video_container = ttk.Frame(main_frame, width=400, height=300)
        video_container.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        video_container.pack_propagate(False)

        self.video_label = ttk.Label(video_container)
        self.video_label.pack()
        
        # Info frame
        info_frame = ttk.LabelFrame(main_frame, text="System Information")
        info_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        


        
        # Status labels
        self.status_var = tk.StringVar(value="Status: Ready")
        ttk.Label(info_frame, textvariable=self.status_var, font=('Arial', 10)).pack(anchor='w', pady=5)
        self.objects_var = tk.StringVar(value="Objects detected: 0")
        ttk.Label(info_frame, textvariable=self.objects_var, font=('Arial', 10)).pack(anchor='w', pady=5)
        self.coords_var = tk.StringVar(value="Object coordinates: None")
        ttk.Label(info_frame, textvariable=self.coords_var, font=('Arial', 10)).pack(anchor='w', pady=5)
        
        
        # Control buttons frame
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=1, column=0, columnspan=2, sticky='ew', pady=10)
        
        # Detection button
        self.detection_btn = ttk.Button(
            btn_frame, 
            text="Enable Detection", 
            command=self.toggle_detection,
            width=15
        )
        self.detection_btn.pack(side='left', padx=5)
        
        # Grab button
        self.grab_btn = ttk.Button(
            btn_frame,
            text="Grab Object",
            command=self.grab_object,
            state='disabled',
            width=15
        )
        self.grab_btn.pack(side='left', padx=5)
        
        # Exit button
        exit_btn = ttk.Button(
            btn_frame,
            text="Exit",
            command=self.on_close,
            width=15
        )
        exit_btn.pack(side='right', padx=5)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
    

    def setup_objects_table(self, parent):
        """Настройка таблицы объектов"""
        self.objects_tree = ttk.Treeview(parent, columns=('ID', 'X', 'Y', 'Width', 'Height'), show='headings', height=5)
        
        # Настройка колонок
        columns = {
            'ID': {'text': 'ID', 'width': 40, 'anchor': 'center'},
            'X': {'text': 'X', 'width': 60, 'anchor': 'center'},
            'Y': {'text': 'Y', 'width': 60, 'anchor': 'center'},
            'Width': {'text': 'Width', 'width': 60, 'anchor': 'center'},
            'Height': {'text': 'Height', 'width': 60, 'anchor': 'center'}
        }
        
        for col, params in columns.items():
            self.objects_tree.heading(col, text=params['text'])
            self.objects_tree.column(col, width=params['width'], anchor=params['anchor'])
        
        # Добавляем прокрутку
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.objects_tree.yview)
        self.objects_tree.configure(yscrollcommand=scrollbar.set)
        
        # Размещаем элементы
        self.objects_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')


    def toggle_detection(self):
        self.vision.detection_enabled = not self.vision.detection_enabled
        state = "ON" if self.vision.detection_enabled else "OFF"
        self.detection_btn.config(text=f"Detection {state}")
        
        if self.vision.detection_enabled:
            self.grab_btn.config(state='normal')
        else:
            self.grab_btn.config(state='disabled')
            self.coords_var.set("Object coordinates: None")
    
    def grab_object(self):
        """Взять  объект"""
    
    def update_video(self):
        ret, frame = self.vision.get_frame()
        
        if ret:
            # Рисуем обнаруженные объекты
            if self.vision.detection_enabled and self.vision.detected_objects:
                for obj in self.vision.detected_objects:
                    x, y = obj['position']
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    cv2.drawContours(frame, [obj['contour']], -1, (0, 255, 0), 2)
                    
                    # Обновляем координаты первого объекта
                    self.coords_var.set(f"Object coordinates: X={x}, Y={y}")
            
            # Конвертируем для Tkinter
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk,)
            
            # Обновляем информацию о количестве объектов
            count = len(self.vision.detected_objects)
            self.objects_var.set(f"Objects detected: {count}")
            
            if count == 0 and self.vision.detection_enabled:
                self.coords_var.set("Object coordinates: None")
        
        self.root.after(30, self.update_video)
    


    def process_detected_objects(self, frame):
        """Обработка обнаруженных объектов"""
        for i, obj in enumerate(self.vision.detected_objects):
            x, y = obj['position']
            cv2.drawContours(frame, [obj['contour']], -1, (0, 255, 0), 1)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            cv2.putText(frame, str(i+1), (x-10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        self.update_objects_table()


    def on_close(self):
        self.vision.release()
        self.root.destroy()