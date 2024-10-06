import tkinter as tk
from tkinter import ttk, messagebox, filedialog, Menu
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk
import os

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

def get_landmarks(image, face_mesh):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0]
    points = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in landmarks.landmark]
    return points

def calculate_delaunay_triangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p)
    triangle_list = subdiv.getTriangleList()
    delaunay_tri = []
    for t in triangle_list:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        idx = []
        for pt in pts:
            for i, p in enumerate(points):
                if abs(pt[0] - p[0]) < 1.0 and abs(pt[1] - p[1]) < 1.0:
                    idx.append(i)
                    break
        if len(idx) == 3:
            delaunay_tri.append(tuple(idx))
    return delaunay_tri

def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def warp_triangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    t1_rect = [(t1[i][0] - r1[0], t1[i][1] - r1[1]) for i in range(3)]
    t2_rect = [(t2[i][0] - r2[0], t2[i][1] - r2[1]) for i in range(3)]
    t2_rect_int = [(int(t2[i][0] - r2[0]), int(t2[i][1] - r2[1])) for i in range(3)]
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)
    img1_rect = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    size = (r2[2], r2[3])
    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)
    img2_rect = img2_rect * mask
    img2_patch = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    if img2_patch.shape == img2_rect.shape:
        img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2_patch * (1 - mask) + img2_rect
    else:
        print(f"Shape mismatch: {img2_patch.shape} vs {img2_rect.shape}")

# Function to read and process the source image
def process_source_image(image_path):
    try:
        pil_image = Image.open(image_path)
    except FileNotFoundError:
        messagebox.showerror("Error", "Source image not found.")
        return None, None, None, None
    source_image = np.array(pil_image)
    source_image = cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR)
    source_landmarks = get_landmarks(source_image, face_mesh)
    if source_landmarks is None:
        messagebox.showerror("Error", "No face detected in the source image.")
        return None, None, None, None
    source_points = np.array(source_landmarks, np.int32)
    source_convexhull = cv2.convexHull(source_points)
    source_rect = cv2.boundingRect(source_convexhull)
    delaunay_tri = calculate_delaunay_triangles(source_rect, source_landmarks)
    if len(delaunay_tri) == 0:
        messagebox.showerror("Error", "No Delaunay triangles found in the source image.")
        return None, None, None, None
    return source_image, source_landmarks, source_points, delaunay_tri

# Read the initial source face image
source_image, source_landmarks, source_points, delaunay_tri = process_source_image("C:/Users/Duane/Desktop/code/images/modi.jpg")

class CustomMenu(Menu):
    def __init__(self, parent, **kwargs):
        Menu.__init__(self, parent, **kwargs)
        self.config(background='black', foreground='white', activebackground='gray', activeforeground='white', tearoff=0)

class FaceSwapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Swap App")
        self.root.geometry("800x600")
        self.root.configure(bg='black')

        # Create a menu bar
        self.menu_bar = CustomMenu(root)
        root.config(menu=self.menu_bar)

        # Add File menu
        file_menu = CustomMenu(self.menu_bar)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Add Image", command=self.add_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=root.quit)

        # Add Help menu
        help_menu = CustomMenu(self.menu_bar)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

        style = ttk.Style()
        style.configure('TButton', font=('Helvetica', 12, 'bold'), background='black', foreground='white')
        style.map('TButton', background=[('active', 'black')], foreground=[('active', 'white')])
        style.configure('TLabel', background='black', foreground='white')

        self.video_label = ttk.Label(root)
        self.video_label.pack(pady=20)

        button_frame = ttk.Frame(root, style='TFrame')
        button_frame.pack(pady=20)

        self.start_button = ttk.Button(button_frame, text="Start Face Swap", command=self.start_face_swap, style='TButton')
        self.start_button.grid(row=0, column=0, padx=10)

        self.stop_button = ttk.Button(button_frame, text="Stop Face Swap", command=self.stop_face_swap, style='TButton')
        self.stop_button.grid(row=0, column=1, padx=10)

        self.add_image_button = ttk.Button(button_frame, text="Add Image", command=self.add_image, style='TButton')
        self.add_image_button.grid(row=0, column=2, padx=10)

        self.webcam_video_stream = cv2.VideoCapture(0)
        self.source_image = source_image
        self.source_landmarks = source_landmarks
        self.source_points = source_points
        self.delaunay_tri = delaunay_tri
        self.running = False

    def start_face_swap(self):
        self.running = True
        self.update_frame()

    def stop_face_swap(self):
        self.running = False

    def add_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.source_image, self.source_landmarks, self.source_points, self.delaunay_tri = process_source_image(file_path)
            if self.source_landmarks is None:
                messagebox.showerror("Error", "No face detected in the selected image.")

    def update_frame(self):
        if not self.running:
            return
        ret, current_frame = self.webcam_video_stream.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            self.root.after(10, self.update_frame)
            return
        destination_image = current_frame.copy()
        destination_image_gray = cv2.cvtColor(destination_image, cv2.COLOR_BGR2GRAY)
        dest_landmarks = get_landmarks(destination_image, face_mesh)
        if dest_landmarks is None:
            self.display_image(destination_image)
            self.root.after(10, self.update_frame)
            return
        dest_points = np.array(dest_landmarks, np.int32)
        dest_convexhull = cv2.convexHull(dest_points)
        warped_face = np.zeros_like(destination_image, dtype=np.float32)
        for tri_indices in self.delaunay_tri:
            t1 = [self.source_landmarks[tri_indices[0]], self.source_landmarks[tri_indices[1]], self.source_landmarks[tri_indices[2]]]
            t2 = [dest_landmarks[tri_indices[0]], dest_landmarks[tri_indices[1]], dest_landmarks[tri_indices[2]]]
            warp_triangle(self.source_image, warped_face, t1, t2)
        warped_face = np.uint8(warped_face)
        mask = np.zeros(destination_image.shape, dtype=destination_image.dtype)
        cv2.fillConvexPoly(mask, dest_convexhull, (255, 255, 255))
        dest_rect = cv2.boundingRect(dest_convexhull)
        dest_center = (int(dest_rect[0] + dest_rect[2] / 2), int(dest_rect[1] + dest_rect[3] / 2))
        try:
            output = cv2.seamlessClone(warped_face, destination_image, mask, dest_center, cv2.NORMAL_CLONE)
        except cv2.error as e:
            print(f"SeamlessClone Error: {e}")
            output = destination_image
        self.display_image(output)
        self.root.after(10, self.update_frame)

    def display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.img_tk = img_tk
        self.video_label.config(image=img_tk)

    def show_about(self):
        messagebox.showinfo("About", "Face Swap App\nDeveloped by Duane Productions")

    def on_closing(self):
        self.webcam_video_stream.release()
        self.root.destroy()

class SplashScreen:
    def __init__(self, root):
        self.root = root
        self.root.title("Loading...")
        self.root.geometry("400x300")
        self.root.configure(bg='black')

        self.label = ttk.Label(root, text="Duane Productions", font=('Helvetica', 24, 'bold'), background='black', foreground='white')
        self.label.pack(expand=True)

        self.root.after(3000, self.destroy_splash)

    def destroy_splash(self):
        self.root.destroy()
        main_app()

def main_app():
    root = tk.Tk()
    app = FaceSwapApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    splash_root = tk.Tk()
    splash = SplashScreen(splash_root)
    splash_root.mainloop()