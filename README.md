# 🖐️ siPakar — Real-Time Hand Gesture Recognition Web App

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black?logo=flask&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Landmarker-0097A7?logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-91.92%25-brightgreen)

**siPakar** adalah aplikasi web berbasis *Artificial Intelligence* dan *Computer Vision* yang mampu mendeteksi serta mengklasifikasikan **13 gestur tangan (*hand gesture*)** secara *real-time* melalui webcam. Proyek ini dirancang untuk mendukung interaksi manusia dan komputer (*Human-Computer Interaction*) tanpa sentuhan fisik dengan latensi rendah (< 100 ms) dan akurasi tinggi.

Proyek ini dikembangkan sebagai Tugas Akhir Mata Kuliah **Pengantar Kecerdasan Buatan** di **Institut Teknologi PLN**.

---

## 📋 Daftar Isi
- [Fitur Utama](#-fitur-utama)
- [Daftar 13 Gestur yang Didukung](#-daftar-13-gestur-yang-didukung)
- [Arsitektur & Cara Kerja Sistem](#️-arsitektur--cara-kerja-sistem)
- [Teknologi yang Digunakan](#️-teknologi-yang-digunakan)
- [Hasil Pengujian & Evaluasi](#-hasil-pengujian--evaluasi)
- [Cara Instalasi & Menjalankan Proyek](#-cara-instalasi--menjalankan-proyek)
- [Dokumentasi Proyek](#-dokumentasi-proyek)
- [Tim Pengembang](#-tim-pengembang)

---

## ✨ Fitur Utama
- **Real-Time Multi-Hand Tracking:** Mendeteksi hingga **2 tangan sekaligus** (tangan kiri dan tangan kanan) secara bersamaan lengkap dengan visualisasi 21 titik koordinat kunci (*3D hand landmarks*) dan garis kerangka (*skeleton*).
- **13 Klasifikasi Gestur Heuristik:** Mengklasifikasikan gerakan jari menggunakan perhitungan geometri koordinat berbasis aturan (*rule-based coordinate geometry*) tanpa memerlukan pelatihan ulang model berat.
- **Handedness & Mirroring Correction:** Koreksi otomatis orientasi tangan kiri (*Left Hand*) dan tangan kanan (*Right Hand*) saat menggunakan mode cermin (*horizontal flip*) pada webcam.
- **Interactive Web Dashboard:** Antarmuka web responsif yang menampilkan *live camera stream*, kartu status gestur terpisah untuk tangan kiri dan kanan, indikator *confidence score* (%), serta tombol kontrol kamera (*Start/Stop*).
- **Low Latency Performance:** Rata-rata waktu inferensi di bawah batas 100 ms sehingga nyaman digunakan secara langsung (*real-time*).

---

## 🤟 Daftar 13 Gestur yang Didukung

| No | Nama Gestur | Deskripsi Gerakan Jari |
| :-: | :--- | :--- |
| 1 | **Thumbs Up** | Ibu jari mengarah ke atas, empat jari lainnya mengepal |
| 2 | **Thumbs Down** | Ibu jari mengarah ke bawah, empat jari lainnya mengepal |
| 3 | **Pointing Up** | Jari telunjuk menunjuk lurus ke atas |
| 4 | **Middle Finger** | Jari tengah mengarah ke atas |
| 5 | **Ring Finger** | Jari manis mengarah ke atas |
| 6 | **Pinky Finger** | Jari kelingking mengarah ke atas |
| 7 | **Fist** | Seluruh jari mengepal tertutup |
| 8 | **Hello** | Telapak tangan terbuka lebar (kelima jari terentang) |
| 9 | **Peace Sign** | Jari telunjuk dan jari tengah membentuk huruf "V" |
| 10 | **I Love You** | Ibu jari, jari telunjuk, dan jari kelingking terbuka |
| 11 | **Metal** | Jari telunjuk dan jari kelingking terbuka ke atas |
| 12 | **Okay** | Ujung ibu jari dan jari telunjuk bersentuhan membentuk lingkaran |
| 13 | **Call Me** | Ibu jari dan jari kelingking terentang menyerupai gagang telepon |

---

## ⚙️ Arsitektur & Cara Kerja Sistem

```text
[Webcam Input] 
      │
      ▼
[OpenCV Pre-processing] ──► Horizontal Flip (Mirror) + Konversi Warna BGR ke RGB
      │
      ▼
[MediaPipe Hand Landmarker] ──► Ekstraksi 21 Titik Koordinat 3D (x, y, z) per Tangan
      │
      ▼
[Heuristic Classifier] ──► Analisis Posisi Relatif Ujung Jari & Sendi + Handedness Correction
      │
      ▼
[Flask Web Server] ──► Streaming Video MJPEG (/video) & REST API JSON (/gesture)
      │
      ▼
[Web Dashboard UI] ──► Live Preview + Status Deteksi Tangan Kiri & Kanan
```

---

## 🛠️ Teknologi yang Digunakan
- **Bahasa Pemrograman:** Python 3.8+
- **Computer Vision & AI:** MediaPipe (`hand_landmarker.task`), OpenCV (`opencv-python`), NumPy
- **Backend Web Server:** Flask
- **Frontend & UI/UX:** HTML5, Tailwind CSS, Vanilla JavaScript (`Fetch API`), Figma

---

## 📊 Hasil Pengujian & Evaluasi

### 1. Akurasi Klasifikasi (*Confusion Matrix* — 260 Sampel)
Pengujian dilakukan sebanyak 20 percobaan untuk masing-masing dari 13 kelas gestur:
- **Rata-rata Akurasi Keseluruhan:** **91,92%**
- **Akurasi Sempurna (100%):** Tercapai pada **8 gestur** (*Pointing Up, Pinky Finger, Fist, Hello, Peace Sign, I Love You, Metal,* dan *Okay*).
- **Gestur Vertikal (*Thumbs Up* & *Thumbs Down*):** Mencapai akurasi **70%** karena sensitivitas sudut kemiringan tangan pada sumbu vertikal ($Y$).

### 2. Performa Waktu Inferensi (*Real-Time Benchmark*)

| Kondisi Pengujian | Rata-rata Inference Time | Frame Rate (FPS) | Status Target (< 100 ms) |
| :--- | :---: | :---: | :---: |
| **Idle (0 Tangan)** | 32,38 ms | 30,8 FPS | ✅ Memenuhi Target |
| **1 Tangan Terdeteksi** | 54,47 ms | 18,6 FPS | ✅ Memenuhi Target |
| **2 Tangan Terdeteksi** | 71,41 ms | 13,9 FPS | ✅ Memenuhi Target |

### 3. Pengujian Fungsional (*Blackbox Testing*)
Seluruh **14 skenario pengujian fungsional** (mulai dari variasi jarak dekat/jauh, perubahan pencahayaan, gerakan tangan cepat, hingga penggunaan berkelanjutan > 5 menit) berjalan stabil tanpa *crash*.

---

## 🚀 Cara Instalasi & Menjalankan Proyek

1. **Clone repositori ini:**
   ```bash
   git clone [https://github.com/bebibulann/hand-gesture-recognition-siPAKAR.git](https://github.com/bebibulann/hand-gesture-recognition-siPAKAR.git)
   cd hand-gesture-recognition-siPAKAR
   ```

2. **Buat dan aktifkan *virtual environment* (opsional namun disarankan):**
   ```bash
   python -m venv venv

   # Windows:
   venv\Scripts\activate

   # macOS / Linux:
   source venv/bin/activate
   ```

3. **Install seluruh *library* yang dibutuhkan:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Pastikan model MediaPipe tersedia:**
   Pastikan file model `hand_landmarker.task` berada di lokasi yang sesuai dengan konfigurasi `model_asset_path` di dalam file `app.py`.

5. **Jalankan aplikasi Flask:**
   ```bash
   python app.py
   ```

6. **Akses aplikasi di browser:**
   Buka browser dan kunjungi alamat:
   ```text
   [http://127.0.0.1:5000](http://127.0.0.1:5000)
   ```
   Klik tombol **Start** pada halaman web untuk mengaktifkan webcam dan mulai mendeteksi gestur tangan.

---

## 📄 Dokumentasi Proyek
Laporan lengkap proyek (analisis kebutuhan, perancangan sistem, implementasi kode, tabel *Confusion Matrix*, serta hasil *Blackbox Testing*) tersedia di dalam repositori ini pada file laporan proyek yang disertakan.

---

## 👥 Tim Pengembang

**Kelompok 4 — SI PAKAR**  
Program Studi S1 Sistem Informasi, Fakultas Telematika Energi  
**Institut Teknologi PLN**

| Nama Lengkap | NIM |
| :--- | :---: |
| **Kamalia Kurniasari** | 202432005 |
| **Trista Yoseva Sitorus** | 202432036 |
| **Ni Nyoman Sutriani** | 202432037 |
| **Petique Bulan Nanindah** | 202432046 |

**Dosen Pengampu:** Dewi Arianti Wulandari, S.Kom., MMSI.
