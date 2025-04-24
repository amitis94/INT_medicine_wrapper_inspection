# 🩺 Medicine Wrapper Machine Vision Inspection

**약 포장지 생산 공정 개선을 위한 실시간 머신비전 검사 시스템**

Line scan camera를 이용해 이미지를 실시간으로 촬영하고, 불량 여부를 판별하며, 불량 객체 정보를 반환하거나 검사 이미지를 저장합니다.

---

## 📚 목차

- [📖 소개](#-소개)
- [✨ 주요 기능](#-주요-기능)
- [🗂 프로젝트 구조](#-프로젝트-구조)
- [⚙️ 설치 방법](#️-설치-방법)

---

## 📖 소개

본 프로젝트는 고속 생산 라인에서 약 포장지의 불량 여부를 자동으로 검사하여 공정 효율을 개선하기 위해 제작되었습니다.

> Line scan 기반 고해상도 영상 분석 + 불량 판단 + API 제공

---

## ✨ 주요 기능

- 🔍 **불량 판별 (OK/NG)**
- 🎯 **불량 객체의 위치 정보 반환**
- 🌐 **FastAPI 기반 API 제공**  *(곽동혁 책임 연구원님의 도움을 받았습니다)*
- 🧵 **2-pixel 단위 Line scan image 검사 구현**

---

## 🗂 프로젝트 구조

INT_medicine_wrapper_inspection
│  .gitignore
│  requirements.txt
│  server.py
│
└─machine_vision
    │  main.py
    │  schemas.py
    │  __init__.py
    │
    ├─images
    │      .DS_Store
    │      golden_template_cam0(20250227)_mean.png
    │      golden_template_cam0(20250227)_median.png
    │      golden_template_cam0(before).bmp
    │      golden_template_cam1(20250227)_mean.png
    │      golden_template_cam1(20250227)_median.png
    │      golden_template_cam1(20250304)_mean_resize.png
    │      golden_template_cam1(before).bmp
    │      golden_template_cam1.bmp
    │
    └─model
            functions.py


## ⚙️ 설치 방법

```bash
git clone https://github.com/your-username/INT_medicine_wrapper_inspection.git
cd INT_medicine_wrapper_inspection