import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import threading
from pathlib import Path

def select_input():
    filename = filedialog.askopenfilename(title="選擇輸入影片", filetypes=[("AVI Video", "*.avi"), ("All Files", "*.*")])
    if filename:
        entry_input.delete(0, tk.END)
        entry_input.insert(0, filename)

def select_output():
    filename = filedialog.asksaveasfilename(title="選擇儲存影片", defaultextension=".avi", filetypes=[("AVI Video", "*.avi")])
    if filename:
        entry_output.delete(0, tk.END)
        entry_output.insert(0, filename)

def select_img():
    filename = filedialog.askopenfilename(title="選擇竄改圖片 (選填)", filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")])
    if filename:
        entry_img.delete(0, tk.END)
        entry_img.insert(0, filename)

def run_script():
    input_video = entry_input.get().strip()
    output_video = entry_output.get().strip()
    img_path = entry_img.get().strip()
    x_val = entry_x.get().strip()
    y_val = entry_y.get().strip()
    scale_val = entry_scale.get().strip()
    start_val = entry_start.get().strip()
    end_val = entry_end.get().strip()
    
    if not input_video:
        messagebox.showerror("錯誤", "必須選擇要被竄改的輸入影片！")
        return
    if not output_video:
        output_video = "tampered.avi"
        
    cmd = [
        "python", "tamper_video.py", 
        "-i", input_video, 
        "-o", output_video, 
        "-x", x_val, 
        "-y", y_val,
        "--scale", scale_val,
        "--start_frame", start_val,
        "--end_frame", end_val
    ]
    
    if img_path:
        cmd.extend(["--img", img_path])
        
    btn_run.config(state=tk.DISABLED, text="處理中，請稍候...")
    
    def process():
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            
            if result.returncode == 0:
                messagebox.showinfo("成功", f"影片局部竄改處理完成！\n已存至：{output_video}")
            else:
                messagebox.showerror("執行失敗", "執行發生錯誤：\n" + result.stderr)
        except Exception as e:
            messagebox.showerror("意外錯誤", f"系統錯誤：{str(e)}\n請檢查是否有安裝相關套件。")
        finally:
            btn_run.config(state=tk.NORMAL, text="開始竄改處理")

    # 開啟多執行緒，避免 GUI 卡死
    threading.Thread(target=process, daemon=True).start()

# --- 建立視窗 ---
root = tk.Tk()
root.title("影片局部竄改測試工具")
root.geometry("600x480")
root.resizable(False, False)

tk.Label(root, text="🎬 影片局部竄改測試介面", font=("微軟正黑體", 16, "bold"), fg="#D32F2F").pack(pady=15)
tk.Label(root, text="可以設定竄改的大小比例與畫格區間，這能幫你驗證哪些片段被變造過！", font=("微軟正黑體", 10)).pack(pady=0)

frame = tk.Frame(root)
frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

# 輸入影片
tk.Label(frame, text="✅ 輸入影片 (.avi):", font=("微軟正黑體", 10)).grid(row=0, column=0, sticky='e', pady=10)
entry_input = tk.Entry(frame, width=40, font=("Arial", 10))
entry_input.grid(row=0, column=1, padx=5)
tk.Button(frame, text="瀏覽...", command=select_input, font=("微軟正黑體", 9)).grid(row=0, column=2)

# 輸出影片
tk.Label(frame, text="💾 輸出影片 (.avi):", font=("微軟正黑體", 10)).grid(row=1, column=0, sticky='e', pady=10)
entry_output = tk.Entry(frame, width=40, font=("Arial", 10))
entry_output.insert(0, "tampered.avi")
entry_output.grid(row=1, column=1, padx=5)
tk.Button(frame, text="瀏覽...", command=select_output, font=("微軟正黑體", 9)).grid(row=1, column=2)

# 竄改用圖片
tk.Label(frame, text="🖼️ 外來圖片 (選填):", font=("微軟正黑體", 10)).grid(row=2, column=0, sticky='e', pady=10)
entry_img = tk.Entry(frame, width=40, font=("Arial", 10))
entry_img.grid(row=2, column=1, padx=5)
tk.Button(frame, text="瀏覽...", command=select_img, font=("微軟正黑體", 9)).grid(row=2, column=2)

# X, Y 座標準位 & 縮放參數
coords_frame = tk.Frame(frame)
coords_frame.grid(row=3, column=1, sticky='w', pady=10)

tk.Label(coords_frame, text="起始 X:", font=("微軟正黑體", 9)).pack(side=tk.LEFT)
entry_x = tk.Entry(coords_frame, width=5, font=("Arial", 10))
entry_x.insert(0, "100")
entry_x.pack(side=tk.LEFT, padx=3)

tk.Label(coords_frame, text="起始 Y:", font=("微軟正黑體", 9)).pack(side=tk.LEFT, padx=(10, 0))
entry_y = tk.Entry(coords_frame, width=5, font=("Arial", 10))
entry_y.insert(0, "100")
entry_y.pack(side=tk.LEFT, padx=3)

tk.Label(coords_frame, text="縮放比例:", font=("微軟正黑體", 9), fg="blue").pack(side=tk.LEFT, padx=(10, 0))
entry_scale = tk.Entry(coords_frame, width=5, font=("Arial", 10))
entry_scale.insert(0, "1.0")
entry_scale.pack(side=tk.LEFT, padx=3)
tk.Label(coords_frame, text="(1.0為原尺寸)", font=("微軟正黑體", 8), fg="gray").pack(side=tk.LEFT)

# 畫格區間設定
frame_range = tk.Frame(frame)
frame_range.grid(row=4, column=1, sticky='w', pady=10)

tk.Label(frame_range, text="起始畫格:", font=("微軟正黑體", 9), fg="green").pack(side=tk.LEFT)
entry_start = tk.Entry(frame_range, width=6, font=("Arial", 10))
entry_start.insert(0, "0")
entry_start.pack(side=tk.LEFT, padx=3)

tk.Label(frame_range, text="結束畫格:", font=("微軟正黑體", 9), fg="green").pack(side=tk.LEFT, padx=(15, 0))
entry_end = tk.Entry(frame_range, width=6, font=("Arial", 10))
entry_end.insert(0, "-1")
entry_end.pack(side=tk.LEFT, padx=3)
tk.Label(frame_range, text="(-1 為到底)", font=("微軟正黑體", 8), fg="gray").pack(side=tk.LEFT)

# 執行按鈕
btn_run = tk.Button(root, text="開始竄改處理", font=("微軟正黑體", 12, "bold"), bg="#D32F2F", fg="white", command=run_script)
btn_run.pack(pady=15, ipadx=20, ipady=5)

root.mainloop()
