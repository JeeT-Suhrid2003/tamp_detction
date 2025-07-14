# Prototype: Secure Image Encryption + Blockchain Hashing (Python Only)

import hashlib
from Crypto.Cipher import AES
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

# 1. Image Encryption (AES ECB)
def pad(data):
    length = 16 - (len(data) % 16)
    return data + bytes([length]) * length

def unpad(data):
    return data[:-data[-1]]

def encrypt_image(image_path, key):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img_data = np.array(img).tobytes()
    cipher = AES.new(key, AES.MODE_ECB)
    padded = pad(img_data)
    encrypted = cipher.encrypt(padded)
    return encrypted, img.size

def decrypt_image(encrypted_data, key, size):
    cipher = AES.new(key, AES.MODE_ECB)
    decrypted = unpad(cipher.decrypt(encrypted_data))
    img_array = np.frombuffer(decrypted, dtype=np.uint8).reshape(size[::-1])
    return Image.fromarray(img_array)

# 2. SHA-256 Hashing
def hash_sha256(data):
    return hashlib.sha256(data).hexdigest()

# 3. Simulate Blockchain Store (local dict)
blockchain_ledger = {}

def store_to_blockchain(image_id, hashed_value):
    blockchain_ledger[image_id] = hashed_value
    print(f"Stored on blockchain: {image_id} -> {hashed_value}")

# 4. Verify Hash
def verify_image(image_path, key, image_id):
    encrypted, _ = encrypt_image(image_path, key)
    current_hash = hash_sha256(encrypted)
    stored_hash = blockchain_ledger.get(image_id)
    return stored_hash == current_hash

# 5. Plot Histograms
def plot_histograms(original_path, encrypted_data):
    img = Image.open(original_path).convert('L')
    plt.figure(figsize=(10, 4))

    # Original image histogram
    plt.subplot(1, 2, 1)
    plt.hist(np.array(img).flatten(), bins=256, range=(0, 256), color='gray')
    plt.title('Original Image Histogram')

    # Encrypted image histogram
    encrypted_np = np.frombuffer(encrypted_data, dtype=np.uint8)
    plt.subplot(1, 2, 2)
    plt.hist(encrypted_np, bins=256, range=(0, 256), color='blue')
    plt.title('Encrypted Image Histogram')

    plt.tight_layout()
    plt.show()

# 6. Add Noise Attack
def apply_noise(image_path, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    noisy = img.copy()
    noise = np.random.randint(0, 2, img.shape).astype(np.uint8) * 255
    noisy = cv2.bitwise_xor(noisy, noise)
    cv2.imwrite(output_path, noisy)
    print(f"Noisy image saved to {output_path}")

# === Main Test ===
if __name__ == "__main__":
    key = b"thisisakey123456"  # Sample 16-byte key
    test_image_path = r"C:\Users\JEET\Pictures\Screenshots\Screenshot 2025-07-13 184245.png"
    
    noisy_image_path = r"C:\Users\JEET\Pictures\Screenshots\Screenshot 2025-07-14 095540.png"
    image_id = "img001"

    encrypted_image, img_size = encrypt_image(test_image_path, key)
    hashed = hash_sha256(encrypted_image)
    store_to_blockchain(image_id, hashed)

    # Decryption Preview
    decrypted_img = decrypt_image(encrypted_image, key, img_size)
    decrypted_img.show(title="Decrypted Image")

    # Histogram comparison
    plot_histograms(test_image_path, encrypted_image)

    # wSimulate tamper with noise
    apply_noise(test_image_path, noisy_image_path)
    is_valid = verify_image(noisy_image_path, key, image_id)
    print("Verification after noise attack:", "Valid" if is_valid else "Tampered")
