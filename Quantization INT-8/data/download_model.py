import urllib.request
import zipfile
import os

# Link đính kèm chính thức từ issue #338 của Espressif esp-who
MODEL_URL = "https://github.com/espressif/esp-who/files/8468302/antispoof.zip"
ZIP_PATH = "antispoof.zip"
EXTRACT_DIR = "extract_tmp"
DEST_FILE = "antispoof.espdl"

def download_model():
    print(f"Đang tải model từ: {MODEL_URL}...")
    try:
        urllib.request.urlretrieve(MODEL_URL, ZIP_PATH)
        print("Đã tải xong ZIP.")
        
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
            
        # Tìm file .espdl trong folder giải nén
        found = False
        for root, dirs, files in os.walk(EXTRACT_DIR):
            for file in files:
                if file.endswith(".espdl"):
                    src = os.path.join(root, file)
                    os.replace(src, DEST_FILE)
                    print(f"Thành công! Đã lấy được file: {DEST_FILE}")
                    found = True
                    break
        
        # Dọn dẹp
        if os.path.exists(ZIP_PATH): os.remove(ZIP_PATH)
        import shutil
        if os.path.exists(EXTRACT_DIR): shutil.rmtree(EXTRACT_DIR)
        
        if not found:
            print("Lỗi: Không tìm thấy file .espdl trong file ZIP!")
            
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    download_model()
