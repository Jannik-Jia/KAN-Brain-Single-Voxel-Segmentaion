#!/usr/bin/env python3
"""
Main Analysis Runner for Brain Segmentation Visualization Toolkit
================================================================

This script automatically runs all visualization and analysis tools
based on configuration settings and detected input files.
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import our visualization modules
from visualization_framework import BrainSegmentationVisualizer
from advanced_visualizations import AdvancedSegmentationAnalyzer
from class_performance_analyzer import ClassPerformanceAnalyzer

class VisualizationController:
    """Main controller for running all visualization analyses"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the controller with configuration"""
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # File paths will be detected/set later
        self.softmax_path = None
        self.info_path = None
        self.label_path = None
        self.output_dir = None
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'paths': {
                'output_dir': 'visualization_results'
            },
            'analysis': {
                'run_basic_visualization': True,
                'run_advanced_analysis': True,
                'run_performance_analysis': True,
                'top_n_classes': 20,
                'slice_selection': 'auto',
                'min_support_threshold': 100
            },
            'visualization': {
                'dpi': 150,
                'figure_format': 'png',
                'generate_html_reports': True
            },
            'logging': {
                'level': 'INFO',
                'save_logs': True,
                'log_file': 'visualization.log'
            }
        }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO').upper())
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # Configure root logger
        logging.basicConfig(
            level=level,
            handlers=[console_handler],
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler if requested
        if log_config.get('save_logs', True):
            log_file = log_config.get('log_file', 'visualization.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)
    
    def detect_input_files(self, search_dirs: List[str] = None) -> bool:
        """
        Auto-detect input files based on naming patterns
        
        Args:
            search_dirs: Directories to search in (defaults to current and results/)
            
        Returns:
            True if all required files found, False otherwise
        """
        if search_dirs is None:
            search_dirs = [".", "results", "../results", "output", "../output"]
        
        self.logger.info("Auto-detecting input files...")
        
        # Patterns to look for
        patterns = {
            'softmax': '*softmax_3d*.nii.gz',
            'info': '*softmax_info*.json',
            'labels': ['*labels_3d*.nii.gz', '*label*.nii.gz', 'balanced_labels*.nii.gz']
        }
        
        found_files = {}
        
        # Search for files
        for search_dir in search_dirs:
            search_path = Path(search_dir)
            if not search_path.exists():
                continue
                
            self.logger.debug(f"Searching in {search_path}...")
            
            # Look for softmax files
            if 'softmax' not in found_files:
                softmax_files = list(search_path.glob(patterns['softmax']))
                if softmax_files:
                    found_files['softmax'] = sorted(softmax_files, key=lambda x: x.stat().st_mtime)[-1]
            
            # Look for info files
            if 'info' not in found_files:
                info_files = list(search_path.glob(patterns['info']))
                if info_files:
                    found_files['info'] = sorted(info_files, key=lambda x: x.stat().st_mtime)[-1]
            
            # Look for label files
            if 'labels' not in found_files:
                for pattern in patterns['labels']:
                    label_files = list(search_path.glob(pattern))
                    if label_files:
                        found_files['labels'] = sorted(label_files, key=lambda x: x.stat().st_mtime)[-1]
                        break
        
        # Set found files
        self.softmax_path = str(found_files.get('softmax', ''))
        self.info_path = str(found_files.get('info', ''))
        self.label_path = str(found_files.get('labels', ''))
        
        # Log results
        self.logger.info("File detection results:")
        self.logger.info(f"  Softmax file: {self.softmax_path or 'NOT FOUND'}")
        self.logger.info(f"  Info file: {self.info_path or 'NOT FOUND'}")
        self.logger.info(f"  Label file: {self.label_path or 'NOT FOUND'}")
        
        # Check if required files are found
        required_found = bool(self.softmax_path and self.info_path)
        if not required_found:
            self.logger.error("Required files (softmax + info) not found!")
        
        return required_found
    
    def set_input_files(self, softmax_path: str, info_path: str = None, 
                       label_path: str = None):
        """
        Manually set input file paths
        
        Args:
            softmax_path: Path to softmax nifti file
            info_path: Path to info json file (auto-detected if None)
            label_path: Path to label file (optional)
        """
        self.softmax_path = softmax_path
        
        # Auto-detect info file if not provided
        if info_path is None:
            softmax_stem = Path(softmax_path).stem.replace('.nii', '')
            info_pattern = softmax_stem.replace('softmax_3d', 'softmax_info') + '.json'
            info_path = str(Path(softmax_path).parent / info_pattern)
            
        self.info_path = info_path
        self.label_path = label_path
        
        # Validate files exist
        if not Path(self.softmax_path).exists():
            raise FileNotFoundError(f"Softmax file not found: {self.softmax_path}")
        if not Path(self.info_path).exists():
            raise FileNotFoundError(f"Info file not found: {self.info_path}")
        if self.label_path and not Path(self.label_path).exists():
            self.logger.warning(f"Label file not found: {self.label_path}")
            self.label_path = None
    
    def setup_output_directory(self) -> Path:
        """Setup and create output directory"""
        output_dir = self.config['paths'].get('output_dir', 'visualization_results')
        
        # Make it timestamp-based for uniqueness
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_dir}_{timestamp}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.output_dir = output_path
        self.logger.info(f"Output directory: {output_path}")
        
        return output_path
    
    def run_basic_visualization(self):
        """Run basic visualization analysis"""
        if not self.config['analysis'].get('run_basic_visualization', True):
            self.logger.info("Skipping basic visualization (disabled in config)")
            return
        
        self.logger.info("=" * 60)
        self.logger.info("RUNNING BASIC VISUALIZATION ANALYSIS")
        self.logger.info("=" * 60)
        
        try:
            visualizer = BrainSegmentationVisualizer(
                self.softmax_path, 
                self.info_path, 
                self.label_path
            )
            
            output_basic = self.output_dir / "basic_visualization"
            output_basic.mkdir(exist_ok=True)
            
            # Run analyses
            self.logger.info("Generating uncertainty analysis...")
            visualizer.visualize_uncertainty_slices(
                save_path=output_basic / f"uncertainty_analysis.{self.config['visualization']['figure_format']}"
            )
            
            self.logger.info("Generating class confidence distribution...")
            visualizer.plot_class_confidence_distribution(
                top_n=self.config['analysis']['top_n_classes'],
                save_path=output_basic / f"confidence_distribution.{self.config['visualization']['figure_format']}"
            )
            
            if self.config['visualization'].get('generate_html_reports', True):
                self.logger.info("Generating interactive 3D browser...")
                visualizer.create_interactive_3d_browser(
                    save_path=output_basic / "interactive_browser.html"
                )
            
            self.logger.info("Analyzing spatial consistency...")
            visualizer.analyze_spatial_consistency(
                save_path=output_basic / f"spatial_consistency.{self.config['visualization']['figure_format']}"
            )
            
            if self.label_path:
                self.logger.info("Comparing with ground truth...")
                visualizer.compare_with_ground_truth(
                    save_path=output_basic / f"ground_truth_comparison.{self.config['visualization']['figure_format']}"
                )
            
            self.logger.info("✅ Basic visualization analysis completed")
            
        except Exception as e:
            self.logger.error(f"Error in basic visualization: {str(e)}")
            raise
    
    def run_advanced_analysis(self):
        """Run advanced analysis"""
        if not self.config['analysis'].get('run_advanced_analysis', True):
            self.logger.info("Skipping advanced analysis (disabled in config)")
            return
        
        self.logger.info("=" * 60)
        self.logger.info("RUNNING ADVANCED ANALYSIS")
        self.logger.info("=" * 60)
        
        try:
            analyzer = AdvancedSegmentationAnalyzer(
                self.softmax_path, 
                self.info_path
            )
            
            output_advanced = self.output_dir / "advanced_analysis"
            analyzer.generate_comprehensive_report(str(output_advanced))
            
            self.logger.info("✅ Advanced analysis completed")
            
        except Exception as e:
            self.logger.error(f"Error in advanced analysis: {str(e)}")
            raise
    
    def run_performance_analysis(self):
        """Run performance analysis (requires label file)"""
        if not self.config['analysis'].get('run_performance_analysis', True):
            self.logger.info("Skipping performance analysis (disabled in config)")
            return
        
        if not self.label_path:
            self.logger.warning("Skipping performance analysis (no label file found)")
            return
        
        self.logger.info("=" * 60)
        self.logger.info("RUNNING PERFORMANCE ANALYSIS")
        self.logger.info("=" * 60)
        
        try:
            analyzer = ClassPerformanceAnalyzer(
                self.softmax_path, 
                self.info_path,
                self.label_path
            )
            
            output_performance = self.output_dir / "performance_analysis"
            analyzer.export_detailed_results(str(output_performance))
            
            self.logger.info("✅ Performance analysis completed")
            
        except Exception as e:
            self.logger.error(f"Error in performance analysis: {str(e)}")
            raise
    
    def generate_summary_report(self):
        """Generate a summary HTML report linking all results"""
        self.logger.info("Generating summary report...")
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Brain Segmentation Analysis Report</title>
    <meta charset="utf-8">
    <style>
        body {{ 
            font-family: 'Segoe UI', Arial, sans-serif; 
            margin: 40px; 
            background-color: #f8f9fa;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{ 
            color: #2c3e50; 
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{ 
            color: #34495e; 
            margin-top: 30px; 
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .section {{ 
            margin-bottom: 40px; 
            padding: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
        }}
        .file-info {{ 
            background: #f1f8ff; 
            padding: 15px; 
            border-radius: 5px; 
            margin: 10px 0;
            font-family: monospace;
        }}
        .link-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .link-item {{
            background: #e8f4fd;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        .link-item a {{
            text-decoration: none;
            color: #2980b9;
            font-weight: 500;
        }}
        .link-item a:hover {{
            color: #1a5490;
            text-decoration: underline;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
            float: right;
        }}
        .status {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .status.success {{ background: #d4edda; color: #155724; }}
        .status.warning {{ background: #fff3cd; color: #856404; }}
        .status.error {{ background: #f8d7da; color: #721c24; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Brain Segmentation Analysis Report</h1>
        <div class="timestamp">Generated: {self._get_timestamp()}</div>
        
        <div class="section">
            <h2>📁 Input Files</h2>
            <div class="file-info">
                <strong>Softmax File:</strong> {Path(self.softmax_path).name}<br>
                <strong>Info File:</strong> {Path(self.info_path).name}<br>
                <strong>Label File:</strong> {Path(self.label_path).name if self.label_path else 'Not provided'}
            </div>
        </div>
        
        <div class="section">
            <h2>🎨 Basic Visualization</h2>
            <span class="status success">✅ Completed</span>
            <div class="link-grid">
                <div class="link-item">
                    <strong>Uncertainty Analysis</strong><br>
                    <a href="basic_visualization/uncertainty_analysis.{self.config['visualization']['figure_format']}">View uncertainty heatmaps and predictions</a>
                </div>
                <div class="link-item">
                    <strong>Class Confidence</strong><br>
                    <a href="basic_visualization/confidence_distribution.{self.config['visualization']['figure_format']}">View confidence distribution analysis</a>
                </div>
                <div class="link-item">
                    <strong>Interactive Browser</strong><br>
                    <a href="basic_visualization/interactive_browser.html">Explore 3D interactive visualization</a>
                </div>
                <div class="link-item">
                    <strong>Spatial Consistency</strong><br>
                    <a href="basic_visualization/spatial_consistency.{self.config['visualization']['figure_format']}">View spatial consistency analysis</a>
                </div>
        """
        
        if self.label_path:
            html_content += f"""
                <div class="link-item">
                    <strong>Ground Truth Comparison</strong><br>
                    <a href="basic_visualization/ground_truth_comparison.{self.config['visualization']['figure_format']}">Compare predictions with ground truth</a>
                </div>
            """
        
        html_content += """
            </div>
        </div>
        
        <div class="section">
            <h2>🔬 Advanced Analysis</h2>
            <span class="status success">✅ Completed</span>
            <div class="link-grid">
                <div class="link-item">
                    <strong>Sampling Distribution</strong><br>
                    <a href="advanced_analysis/sampling_distribution.png">View spatial sampling patterns</a>
                </div>
                <div class="link-item">
                    <strong>Class Relationships</strong><br>
                    <a href="advanced_analysis/class_relationships.png">Explore inter-class relationships</a>
                </div>
                <div class="link-item">
                    <strong>Uncertainty Regions</strong><br>
                    <a href="advanced_analysis/uncertainty_analysis.png">Deep dive into uncertainty patterns</a>
                </div>
                <div class="link-item">
                    <strong>Complete Report</strong><br>
                    <a href="advanced_analysis/report.html">View comprehensive HTML report</a>
                </div>
            </div>
        </div>
        """
        
        if self.label_path:
            html_content += f"""
        <div class="section">
            <h2>📊 Performance Analysis</h2>
            <span class="status success">✅ Completed</span>
            <div class="link-grid">
                <div class="link-item">
                    <strong>Performance Metrics</strong><br>
                    <a href="performance_analysis/performance_metrics.csv">Download detailed metrics (CSV)</a>
                </div>
                <div class="link-item">
                    <strong>Confusion Matrix</strong><br>
                    <a href="performance_analysis/confusion_matrix.png">View classification confusion patterns</a>
                </div>
                <div class="link-item">
                    <strong>Class Performance</strong><br>
                    <a href="performance_analysis/class_performance.png">Compare class-level performance</a>
                </div>
                <div class="link-item">
                    <strong>Difficult Regions</strong><br>
                    <a href="performance_analysis/difficult_regions.png">Identify problematic areas</a>
                </div>
                <div class="link-item">
                    <strong>Interactive Dashboard</strong><br>
                    <a href="performance_analysis/interactive_dashboard.html">Explore performance interactively</a>
                </div>
                <div class="link-item">
                    <strong>Summary Report</strong><br>
                    <a href="performance_analysis/summary.json">View performance summary (JSON)</a>
                </div>
            </div>
        </div>
            """
        else:
            html_content += """
        <div class="section">
            <h2>📊 Performance Analysis</h2>
            <span class="status warning">⚠️ Skipped (no label file provided)</span>
            <p>To enable performance analysis, provide a ground truth label file.</p>
        </div>
            """
        
        html_content += """
        <div class="section">
            <h2>ℹ️ How to Use This Report</h2>
            <ul>
                <li><strong>Basic Visualization:</strong> Start here for an overview of your segmentation results</li>
                <li><strong>Advanced Analysis:</strong> Dive deeper into sampling patterns and class relationships</li>
                <li><strong>Performance Analysis:</strong> Quantitative evaluation against ground truth (if available)</li>
                <li><strong>Interactive Elements:</strong> HTML files can be opened in any web browser for interactive exploration</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>🛠️ Technical Details</h2>
            <p><strong>Analysis Framework:</strong> Brain Segmentation Visualization Toolkit</p>
            <p><strong>Generated by:</strong> main_analyzer.py</p>
            <p><strong>Configuration:</strong> config.yaml</p>
        </div>
    </div>
</body>
</html>
        """
        
        summary_path = self.output_dir / "index.html"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"📋 Summary report generated: {summary_path}")
    
    def _get_timestamp(self) -> str:
        """Get formatted timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def run_full_analysis(self):
        """Run all analyses in sequence"""
        self.logger.info("🚀 Starting comprehensive brain segmentation analysis...")
        
        # Setup output directory
        self.setup_output_directory()
        
        try:
            # Run analyses in order
            self.run_basic_visualization()
            self.run_advanced_analysis()
            self.run_performance_analysis()
            
            # Generate summary
            self.generate_summary_report()
            
            self.logger.info("=" * 60)
            self.logger.info("🎉 ALL ANALYSES COMPLETED SUCCESSFULLY!")
            self.logger.info(f"📁 Results saved to: {self.output_dir}")
            self.logger.info(f"📋 Open {self.output_dir}/index.html to view the report")
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"❌ Analysis failed: {str(e)}")
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Brain Segmentation Visualization Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect files and run all analyses
  python main_analyzer.py
  
  # Specify softmax file explicitly
  python main_analyzer.py -s results/test_softmax_3d_subject_bg_excl_20250827.nii.gz
  
  # Specify all files
  python main_analyzer.py -s softmax.nii.gz -i info.json -l labels.nii.gz
  
  # Use custom config
  python main_analyzer.py -c my_config.yaml
        """
    )
    
    parser.add_argument('-s', '--softmax', type=str,
                       help='Path to softmax nifti file')
    parser.add_argument('-i', '--info', type=str,
                       help='Path to info JSON file (auto-detected if not provided)')
    parser.add_argument('-l', '--labels', type=str,
                       help='Path to ground truth labels file (optional)')
    parser.add_argument('-c', '--config', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--search-dirs', nargs='+', default=None,
                       help='Directories to search for input files')
    
    args = parser.parse_args()
    
    # Initialize controller
    controller = VisualizationController(args.config)
    
    try:
        # Set input files
        if args.softmax:
            controller.set_input_files(args.softmax, args.info, args.labels)
        else:
            # Auto-detect files
            if not controller.detect_input_files(args.search_dirs):
                print("❌ Could not find required input files!")
                print("Please specify --softmax path or ensure files are in searchable directories")
                sys.exit(1)
        
        # Run full analysis
        controller.run_full_analysis()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()