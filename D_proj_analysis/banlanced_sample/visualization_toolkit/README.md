# 🧠 Brain Segmentation Visualization Toolkit

A comprehensive analysis and visualization suite for brain segmentation results, specifically designed for cluster-sampled softmax predictions.

## ✨ Features

### 🎨 **Basic Visualization**
- **Uncertainty Analysis**: Entropy-based uncertainty heatmaps and 3D visualizations
- **Class Confidence Distribution**: Statistical analysis of prediction confidence across classes
- **Interactive 3D Browser**: Web-based interactive slice viewer with probability thresholds
- **Spatial Consistency**: Local neighborhood consistency analysis
- **Ground Truth Comparison**: Side-by-side prediction vs truth analysis (when labels available)

### 🔬 **Advanced Analysis**
- **Sampling Distribution**: Spatial analysis of cluster sampling patterns
- **Class Relationship Mining**: Co-occurrence matrices, hierarchical clustering, and transition analysis
- **Probability Landscape**: 3D probability clouds and maximum intensity projections
- **Uncertainty Region Analysis**: Deep dive into prediction uncertainty patterns

### 📊 **Performance Analysis** (requires ground truth labels)
- **Per-Class Metrics**: Precision, Recall, F1-Score, IoU, Dice coefficient for each anatomical structure
- **Confusion Matrix**: Interactive and static visualization of classification errors
- **Difficult Region Identification**: Spatial analysis of error patterns and boundary effects
- **Interactive Performance Dashboard**: Web-based exploration of performance metrics

## 🚀 Quick Start

### **One-Click Analysis**
```bash
cd visualization_toolkit/
./run_visualization.sh
```

The script will automatically:
1. Detect your latest softmax prediction files
2. Install missing Python dependencies
3. Run all analyses
4. Generate an HTML report with all results

### **Specify Files Manually**
```bash
# With specific softmax file
./run_visualization.sh path/to/your_softmax.nii.gz

# With labels for performance analysis
./run_visualization.sh -s softmax.nii.gz -l labels.nii.gz
```

### **Command Line Options**
```bash
./run_visualization.sh [OPTIONS]

Options:
  -s, --softmax FILE    Softmax prediction file (.nii.gz)
  -i, --info FILE       Info JSON file (auto-detected)
  -l, --labels FILE     Ground truth labels (enables performance analysis)
  -c, --config FILE     Configuration file (default: config.yaml)
  -h, --help           Show help message
```

## 📁 File Structure

```
visualization_toolkit/
├── run_visualization.sh          # One-click runner script
├── main_analyzer.py              # Main analysis controller
├── config.yaml                   # Configuration file
├── visualization_framework.py     # Basic visualization tools
├── advanced_visualizations.py    # Advanced analysis tools
├── class_performance_analyzer.py # Performance evaluation tools
└── README.md                     # This file
```

## 📋 Requirements

### **Python Packages**
The script will automatically install missing packages:
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `nibabel` (for NIfTI file handling)
- `scikit-learn`, `scipy` (for metrics and analysis)
- `plotly` (for interactive visualizations)
- `pyyaml` (for configuration)

### **Input Files**
- **Required**: `*softmax_3d*.nii.gz` - Your model's softmax predictions
- **Required**: `*softmax_info*.json` - Metadata file with class information
- **Optional**: `*labels*.nii.gz` - Ground truth labels (enables performance analysis)

## 🎯 File Detection

The toolkit automatically searches for files matching these patterns:
- **Softmax files**: `*softmax_3d*.nii.gz`
- **Info files**: `*softmax_info*.json`
- **Label files**: `*labels*.nii.gz`, `*label*.nii.gz`, `balanced_labels*.nii.gz`

Search locations:
- Current directory
- `results/`, `../results/`
- `output/`, `../output/`
- `banlanced/`, `../banlanced/`
- `logs/`, `../logs/`

## ⚙️ Configuration

Edit `config.yaml` to customize analysis:

```yaml
analysis:
  run_basic_visualization: true
  run_advanced_analysis: true
  run_performance_analysis: true
  top_n_classes: 20
  
visualization:
  dpi: 150
  figure_format: "png"  # "png", "pdf", "svg"
  generate_html_reports: true
```

## 📊 Output Structure

After running analysis, you'll get:

```
visualization_results_TIMESTAMP/
├── index.html                    # Main summary report
├── basic_visualization/
│   ├── uncertainty_analysis.png
│   ├── confidence_distribution.png
│   ├── interactive_browser.html
│   └── spatial_consistency.png
├── advanced_analysis/
│   ├── sampling_distribution.png
│   ├── class_relationships.png
│   ├── uncertainty_analysis.png
│   └── report.html
└── performance_analysis/         # Only if labels provided
    ├── performance_metrics.csv
    ├── confusion_matrix.png
    ├── class_performance.png
    ├── difficult_regions.png
    ├── interactive_dashboard.html
    └── summary.json
```

## 🎨 Visualization Examples

### **Uncertainty Analysis**
![Uncertainty Example](https://via.placeholder.com/800x400/FF6B6B/FFFFFF?text=Uncertainty+Heatmaps+%26+Predictions)

### **Class Performance Matrix**
![Performance Example](https://via.placeholder.com/800x400/4ECDC4/FFFFFF?text=Per-Class+Performance+Metrics)

### **Interactive 3D Browser**
![Interactive Example](https://via.placeholder.com/800x400/45B7D1/FFFFFF?text=Interactive+3D+Slice+Browser)

## 💡 Use Cases

### **Model Development**
- Identify which anatomical structures are hardest to segment
- Understand model uncertainty patterns
- Debug sampling strategy effects

### **Quality Assessment**
- Quantify segmentation performance per structure
- Find spatial error patterns
- Compare different model versions

### **Publication & Presentation**
- Generate publication-ready figures
- Create interactive demonstrations
- Export detailed performance tables

## 🔧 Advanced Usage

### **Python API**
```python
from visualization_framework import BrainSegmentationVisualizer
from class_performance_analyzer import ClassPerformanceAnalyzer

# Basic visualization
viz = BrainSegmentationVisualizer(softmax_path, info_path, label_path)
viz.visualize_uncertainty_slices()
viz.create_interactive_3d_browser()

# Performance analysis
analyzer = ClassPerformanceAnalyzer(softmax_path, info_path, label_path)
df = analyzer.generate_performance_report()
analyzer.export_detailed_results()
```

### **Custom Configuration**
```python
from main_analyzer import VisualizationController

controller = VisualizationController("my_config.yaml")
controller.set_input_files(softmax_path, info_path, label_path)
controller.run_full_analysis()
```

## 🐛 Troubleshooting

### **Common Issues**

1. **Files not found**
   ```bash
   # Check file patterns
   ls -la *softmax*.nii.gz
   ls -la *info*.json
   
   # Specify files manually
   ./run_visualization.sh -s exact/path/to/softmax.nii.gz
   ```

2. **Missing packages**
   ```bash
   # Install manually if auto-install fails
   pip install numpy pandas matplotlib seaborn nibabel scikit-learn scipy plotly pyyaml
   ```

3. **Memory issues**
   ```bash
   # Reduce memory usage in config.yaml
   advanced:
     max_voxels_for_3d_scatter: 500
     cooccurrence_sample_size: 500
   ```

4. **Permission errors**
   ```bash
   chmod +x run_visualization.sh
   ```

### **Getting Help**

- Check the generated `visualization.log` file for detailed error messages
- Ensure input files are valid NIfTI format: `nibabel.load(filename)`
- Verify Python environment: `python3 -c "import nibabel, sklearn, plotly"`

## 📈 Performance Tips

- **Large datasets**: Enable batch processing in config
- **Multiple analyses**: Run analyses separately using Python API
- **Custom visualizations**: Extend the analyzer classes
- **High-resolution figures**: Increase DPI in config

## 🤝 Contributing

This toolkit is designed to be extensible. To add new analyses:

1. Create a new analyzer class following existing patterns
2. Add configuration options in `config.yaml`
3. Integrate with `main_analyzer.py`
4. Update documentation

## 📄 License

Academic and research use. Please cite if used in publications.

## 🙋 Support

For issues with:
- **File formats**: Check NIfTI file integrity with `nibabel`
- **Visualization errors**: Verify matplotlib/plotly installation
- **Performance issues**: Review memory usage and sampling settings
- **Custom modifications**: Refer to source code documentation

---

**Happy analyzing! 🧠✨**