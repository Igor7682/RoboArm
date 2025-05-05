import tkinter as tk
from tkinter import *
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
        self.coords_var = tk.StringVar(value="")
        ttk.Label(info_frame, textvariable=self.coords_var, font=('Arial', 10)).pack(anchor='w', pady=5)
        
        

        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=3, column=0, columnspan=2, sticky='ew', pady=10)
        

        self.detection_btn = ttk.Button(
            btn_frame, 
            text="Enable Detection", 
            command=self.toggle_detection,
            width=15
        )
        self.detection_btn.pack(side='left', padx=5)
        

        self.grab_btn = ttk.Button(
            btn_frame,
            text="Grab Object",
            command=self.grab_object,
            state='disabled',
            width=15
        )
        self.grab_btn.pack(side='left', padx=5)



        self.screen_btn = ttk.Button(
            btn_frame,
            text="Screen save",
            command=self.vision.saveFrame(),
            width=15
        )
        self.screen_btn.pack(side='left', padx=5)
        

        exit_btn = ttk.Button(
            btn_frame,
            text="Exit",
            command=self.on_close,
            width=15
        )
        exit_btn.pack(side='right', padx=5)
        

        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        
        table_frame = ttk.LabelFrame(main_frame, text="Detected Objects")
        table_frame.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)
    
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=2)

        table_frame1 = ttk.LabelFrame(main_frame, text="Detected Objects")
        table_frame1.grid(row=2, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)
    
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=2)
        

        self.setup_objects_table(table_frame)

    

    def setup_objects_table(self, parent):
        self.objects_tree = ttk.Treeview(parent, columns=('ID',"X","Y","W","H"), show='headings', height=2)
        

        columns = {
            'ID': {'text': 'ID', 'width': 40, 'anchor': 'center'},
            'X': {'text': 'X', 'width': 40, 'anchor': 'center'},
            'Y': {'text': 'Y', 'width': 40, 'anchor': 'center'},
            'W': {'text': 'W', 'width': 40, 'anchor': 'center'},
            'H': {'text': 'H', 'width': 40, 'anchor': 'center'},
        }
        
        for col, params in columns.items():
            self.objects_tree.heading(col, text=params['text'])
            self.objects_tree.column(col, width=params['width'], anchor=params['anchor'])
        
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.objects_tree.yview)
        self.objects_tree.configure(yscrollcommand=scrollbar.set)
    

      
        self.objects_tree.pack(side='left', fill='both', expand=True)

        scrollbar.pack(side='right', fill='y')




        self.objects_tree1 = ttk.Treeview(parent, columns=('1'), show='headings', height=2)
        
        columns = {
            '1': {'text': 'Значения углов суставов', 'width': 40, 'anchor': 'center'},
        }
        
        for col, params in columns.items():
            self.objects_tree1.heading(col, text=params['text'])
            self.objects_tree1.column(col, width=params['width'], anchor=params['anchor'])
        
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.objects_tree.yview)
        self.objects_tree1.configure(yscrollcommand=scrollbar.set)
    

      
        self.objects_tree1.pack(side='left', fill='both', expand=True)

        scrollbar.pack(side='right', fill='y')

    def fillTable(self):
        ob = self.vision.getObj()
        for obj in ob:
            self.objects_tree.insert("", END, values=obj)
        pos = self.vision.getPos()
        if len(pos) > 0:
            self.objects_tree1.insert("", END, values=pos)
            print(pos)
    

    def toggle_detection(self):
        self.vision.detection_enabled = not self.vision.detection_enabled
        state = "ON" if self.vision.detection_enabled else "OFF"
        self.detection_btn.config(text=f"Detection {state}")
        
        if self.vision.detection_enabled:
            self.grab_btn.config(state='normal')
            
            
       # else:
            #self.grab_btn.config(state='disabled')
            #self.coords_var.set("Object coordinates: None")
    
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
                    
            # Конвертируем для Tkinter
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk,)
            
            # Обновляем информацию о количестве объектов
            count = len(self.vision.detected_objects)
            self.objects_var.set(f"Objects detected: {count}")
            self.objects_tree.delete(*self.objects_tree.get_children())
            self.objects_tree1.delete(*self.objects_tree1.get_children())
            self.fillTable()
             
        self.root.after(30, self.update_video)
    
    def on_close(self):
        self.vision.release()
        self.root.destroy()