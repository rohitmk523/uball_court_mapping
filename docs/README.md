# Basketball Court Tracking System - Documentation

Complete technical documentation for the Basketball Court Tracking System.

## 📚 Documentation Overview

This documentation suite provides comprehensive coverage of the system from architecture to usage to future improvements.

---

## 📖 Document Index

### 1. [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)
**Complete system design and architecture**

**Contents**:
- System overview and capabilities
- Component architecture (Backend, Frontend, Data Storage)
- Data flow diagrams
- Technology stack
- Design patterns
- Directory structure
- Deployment architecture

**Read this if**: You want to understand how the system is built and how components interact.

**Target Audience**: Developers, architects, technical leads

---

### 2. [API_REFERENCE.md](API_REFERENCE.md)
**Complete REST API documentation**

**Contents**:
- Tags API (UWB data operations)
- Court API (geometry and rendering)
- Calibration API (homography computation)
- Tracking API (video processing and matching)
- Error handling
- Data models
- Code examples (curl, Python, JavaScript)

**Read this if**: You need to integrate with the API or understand endpoint functionality.

**Target Audience**: API consumers, integration developers, testers

---

### 3. [USER_GUIDE.md](USER_GUIDE.md)
**Step-by-step usage instructions**

**Contents**:
- Getting started
- **Workflow 1**: Visualizing UWB tag data
- **Workflow 2**: Calibrating video with court
- **Workflow 3**: Tracking and matching
- Advanced features
- Troubleshooting
- Tips and best practices

**Read this if**: You want to use the system and learn how to perform each workflow.

**Target Audience**: End users, analysts, operators

---

### 4. [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md)
**In-depth technical implementation**

**Contents**:
- Coordinate systems (Court, Video, Canvas)
- Homography transformation (RANSAC algorithm)
- YOLOv11 detection (bounding boxes, confidence)
- ByteTrack tracking (IoU matching, Hungarian algorithm)
- Tag-to-player matching algorithm
- Timestamp synchronization
- Performance optimization strategies
- Data formats
- Mathematical foundations

**Read this if**: You need to understand the algorithms and mathematics behind the system.

**Target Audience**: Researchers, algorithm developers, advanced users

---

### 5. [FUTURE_IMPROVEMENTS.md](FUTURE_IMPROVEMENTS.md)
**Enhancement roadmap and feature suggestions**

**Contents**:
- High priority improvements (video frame display, progress updates)
- User experience enhancements (validation, tutorials, shortcuts)
- Performance optimizations (caching, database backend)
- Advanced features (multi-camera, re-identification, play classification)
- Data analysis & visualization (statistics, heat maps, exports)
- System architecture upgrades (microservices, containerization)
- Research features (ball tracking, pose estimation, event detection)
- Implementation priority phases

**Read this if**: You want to contribute to the project or understand future direction.

**Target Audience**: Contributors, project managers, stakeholders

---

## 🚀 Quick Start

### For End Users
1. Start with [USER_GUIDE.md](USER_GUIDE.md)
2. Follow the three main workflows
3. Refer to troubleshooting if needed

### For Developers
1. Read [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) for overview
2. Check [API_REFERENCE.md](API_REFERENCE.md) for endpoints
3. Study [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md) for algorithms

### For Contributors
1. Understand [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)
2. Review [FUTURE_IMPROVEMENTS.md](FUTURE_IMPROVEMENTS.md)
3. Choose a feature from the roadmap

---

## 📋 Document Comparison

| Document | Length | Complexity | Audience | Purpose |
|----------|--------|------------|----------|---------|
| SYSTEM_ARCHITECTURE | ~400 lines | Medium | Developers | Understand system design |
| API_REFERENCE | ~800 lines | Low-Medium | API users | API integration |
| USER_GUIDE | ~600 lines | Low | End users | Learn how to use |
| TECHNICAL_DETAILS | ~700 lines | High | Researchers | Deep technical understanding |
| FUTURE_IMPROVEMENTS | ~900 lines | Medium | Contributors | Future development |

---

## 🎯 Common Questions

### "I'm new to the system. Where do I start?"
→ Read [USER_GUIDE.md](USER_GUIDE.md) sections 1-2 (Getting Started, Workflow 1)

### "How do I call the API to get tag data?"
→ See [API_REFERENCE.md](API_REFERENCE.md) → Tags API → Get Tag Data

### "What algorithm is used for player detection?"
→ Read [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md) → YOLOv11 Detection

### "How does calibration work mathematically?"
→ Read [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md) → Homography Transformation

### "I want to add video frame display. How do I start?"
→ Check [FUTURE_IMPROVEMENTS.md](FUTURE_IMPROVEMENTS.md) → High Priority → #1 Video Frame Display

### "What's the system architecture?"
→ Read [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) → System Components + Architecture Diagrams

### "How do I troubleshoot calibration issues?"
→ See [USER_GUIDE.md](USER_GUIDE.md) → Troubleshooting → Calibration Issues

### "What data formats does the system use?"
→ See [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md) → Data Formats

---

## 📊 System Capabilities Summary

### Current Features ✅
- **UWB Tag Processing**: Parse and visualize 1.3M position records from 19 tags
- **Court Rendering**: Extract and render basketball court from DXF files
- **Tag Animation**: Real-time playback with adjustable FPS (1-30)
- **Manual Calibration**: Homography computation from correspondence points
- **Player Detection**: YOLOv11-based person detection in video
- **Multi-Object Tracking**: ByteTrack for consistent player IDs
- **Tag Matching**: Associate UWB tags with detected players (200cm threshold)
- **Timestamp Sync**: Manual alignment of video and UWB timelines
- **Interactive UI**: Three web pages (Tag Viz, Calibration, Tracking)
- **REST API**: Complete API for all operations
- **Caching**: YOLO detection results cached for performance

### Planned Features ⬜
- Video frame display in tracking UI
- Real-time processing progress
- Automatic calibration with line detection
- Heat maps and trajectory visualization
- Statistics dashboard
- Multi-camera support
- Database backend (PostgreSQL)
- Export to CSV/JSON/Video/PDF
- Ball tracking
- Pose estimation
- Event detection

---

## 🛠️ Technology Stack

**Backend**:
- Python 3.12
- FastAPI 0.109.0
- OpenCV 4.9.0.80
- Ultralytics 8.1.0 (YOLOv11)
- ezdxf 1.2.0
- NumPy 1.26.0
- lapx 0.9.2 (ByteTrack)

**Frontend**:
- Vanilla JavaScript (ES6+)
- HTML5 Canvas
- CSS3 with animations
- Fetch API

**Data Storage**:
- File-based (JSON)
- Future: PostgreSQL

**Deployment**:
- Uvicorn ASGI server
- Conda environment management

---

## 📝 Usage Example

```bash
# 1. Start server
./start_server.sh

# 2. Open browser
open http://localhost:8000

# 3. Process UWB data (via UI button)
# Click "Process Log File"

# 4. Calibrate (via /calibration page)
# Select 10-15 correspondence points

# 5. Process video (via /tracking page)
# Set sync point → Process video → Play back
```

---

## 🔍 Search by Topic

### Coordinate Systems
- [TECHNICAL_DETAILS.md → Coordinate Systems](TECHNICAL_DETAILS.md#coordinate-systems)

### Homography & Calibration
- [TECHNICAL_DETAILS.md → Homography Transformation](TECHNICAL_DETAILS.md#homography-transformation)
- [USER_GUIDE.md → Workflow 2](USER_GUIDE.md#workflow-2-calibrating-video-with-court)
- [API_REFERENCE.md → Calibration API](API_REFERENCE.md#calibration-api)

### YOLO Detection
- [TECHNICAL_DETAILS.md → YOLOv11 Detection](TECHNICAL_DETAILS.md#yolov11-detection)
- [FUTURE_IMPROVEMENTS.md → Different YOLO Models](FUTURE_IMPROVEMENTS.md#using-different-yolo-models)

### Tracking & Matching
- [TECHNICAL_DETAILS.md → ByteTrack Tracking](TECHNICAL_DETAILS.md#bytetrack-tracking)
- [TECHNICAL_DETAILS.md → Tag-to-Player Matching](TECHNICAL_DETAILS.md#tag-to-player-matching)
- [USER_GUIDE.md → Workflow 3](USER_GUIDE.md#workflow-3-tracking-and-matching)

### API Usage
- [API_REFERENCE.md](API_REFERENCE.md) (entire document)
- [SYSTEM_ARCHITECTURE.md → Data Flow](SYSTEM_ARCHITECTURE.md#data-flow)

### Performance
- [TECHNICAL_DETAILS.md → Performance Optimization](TECHNICAL_DETAILS.md#performance-optimization)
- [FUTURE_IMPROVEMENTS.md → Performance Optimizations](FUTURE_IMPROVEMENTS.md#performance-optimizations)

### Troubleshooting
- [USER_GUIDE.md → Troubleshooting](USER_GUIDE.md#troubleshooting)

### Future Features
- [FUTURE_IMPROVEMENTS.md](FUTURE_IMPROVEMENTS.md) (entire document)

---

## 📦 Data Files

### Input Files
```
logs/session_1763778442483.log     - UWB log (1.3M records)
dxf/court_2.dxf                    - Court DXF file
GX020018.MP4                       - Video file (11GB)
```

### Generated Files
```
data/output/tags/                   - Tag JSON files (19 files)
data/calibration/court_image.png    - Rendered court (191KB)
data/calibration/calibration.json   - Homography matrix
data/calibration/sync_point.json    - Sync point data
data/cache/detections/              - YOLO detection cache
```

---

## 🔗 Related Resources

**Project Files**:
- [README.md](../README.md) - Quick start guide
- [TEST_REPORT.md](../TEST_REPORT.md) - End-to-end testing results
- [requirements.txt](../requirements.txt) - Python dependencies
- [environment.yml](../environment.yml) - Conda environment

**External Documentation**:
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Ultralytics YOLOv11 Docs](https://docs.ultralytics.com/)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

---

## 🤝 Contributing

To contribute to the project:

1. **Read Documentation**:
   - [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) - Understand the system
   - [FUTURE_IMPROVEMENTS.md](FUTURE_IMPROVEMENTS.md) - Choose a feature

2. **Setup Development Environment**:
   ```bash
   conda create -n court_tracking python=3.12
   conda activate court_tracking
   pip install -r requirements.txt
   ```

3. **Development Workflow**:
   - Create feature branch
   - Implement feature
   - Test thoroughly
   - Update documentation
   - Submit pull request

4. **Code Standards**:
   - Follow existing code style
   - Add docstrings to functions
   - Include type hints
   - Write tests where applicable

---

## 📄 License

Internal use - Cellstrat

---

## 📧 Contact

For questions, issues, or contributions:
- Open a GitHub issue
- Contact the development team

---

## 📅 Document Version

**Last Updated**: November 27, 2025
**Documentation Version**: 1.0.0
**System Version**: 1.0.0

---

## 🎓 Learning Path

### Beginner
1. [USER_GUIDE.md](USER_GUIDE.md) → Getting Started
2. [USER_GUIDE.md](USER_GUIDE.md) → Workflow 1 (Tag Visualization)
3. [USER_GUIDE.md](USER_GUIDE.md) → Workflow 2 (Calibration)

### Intermediate
4. [API_REFERENCE.md](API_REFERENCE.md) → Tags API
5. [API_REFERENCE.md](API_REFERENCE.md) → Calibration API
6. [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) → System Components

### Advanced
7. [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md) → Homography
8. [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md) → YOLO & ByteTrack
9. [FUTURE_IMPROVEMENTS.md](FUTURE_IMPROVEMENTS.md) → Advanced Features

---

Happy tracking! 🏀
