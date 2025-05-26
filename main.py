import tkinter as tk
from tkinter import ttk
from gui import GraspingGUI
from vision_system import VisionSystem
from robot_arm import RobotArm



def main():
    try:
        # Инициализация систем
        vision_system = VisionSystem()
        robot_arm = RobotArm()
        
        # Создание интерфейса
        root = tk.Tk()
        root.geometry("1280x900")
        
        # Стиль для кнопок
        style = ttk.Style()
        style.configure('TButton', font=('Arial', 10), padding=5)
        style.configure('TLabel', font=('Arial', 10))
        
        app = GraspingGUI(root, vision_system, robot_arm)
        
        # Обработка закрытия окна
        root.protocol("WM_DELETE_WINDOW", app.on_close)
        
        # Запуск главного цикла
        root.mainloop()
        
    except Exception as e:
        print(f"Application error: {str(e)}")
    finally:
        if 'vision_system' in locals():
            vision_system.release()

if __name__ == "__main__":
    main()