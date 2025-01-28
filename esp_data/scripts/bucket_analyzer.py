import re
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Set, Tuple, NamedTuple
from pathlib import Path
from cloudpathlib import CloudPath, GSPath

class FileAnalysis(NamedTuple):
    """Container for individual file analysis results"""
    extension: str
    size_bytes: int
    is_readme: bool
    has_doc_in_name: bool
    is_versioned: bool
    name_length: int
    has_date: bool
    relative_path: str

class GCSBucketAnalyzer:
    def __init__(self, bucket_path: str):
        """Initialize analyzer with a GCS bucket path."""
        self.bucket_path = GSPath(bucket_path)
        self.version_patterns = [
            r'v\d+',                    # matches v1, v2, etc.
            r'v\d+\.\d+',              # matches v1.0, v2.1, etc.
            r'v\d+\.\d+\.\d+'          # matches v1.0.0, v2.1.1, etc.
        ]
        self.date_patterns = [
            r'\d{4}-\d{2}-\d{2}',      # YYYY-MM-DD
            r'\d{8}',                   # YYYYMMDD
            r'\d{2}-\d{2}-\d{4}',      # DD-MM-YYYY
            r'\d{2}_\d{2}_\d{4}'       # DD_MM_YYYY
        ]
    
    def analyze_file(self, file_path: GSPath) -> FileAnalysis:
        """Analyze a single file and return its characteristics."""
        relative_path = str(file_path.relative_to(self.bucket_path))
        name = file_path.name
        
        # Basic file properties
        extension = file_path.suffix.lower() if file_path.suffix else 'no_extension'
        size_bytes = file_path.stat().st_size
        name_length = len(name)
        
        # File name checks
        is_readme = name.lower() == 'readme' or file_path.stem.lower() == 'readme'
        has_doc_in_name = 'doc' in name.lower()
        is_versioned = any(re.search(pattern, name) for pattern in self.version_patterns)
        has_date = any(re.search(pattern, name) for pattern in self.date_patterns)
        
        return FileAnalysis(
            extension=extension,
            size_bytes=size_bytes,
            is_readme=is_readme,
            has_doc_in_name=has_doc_in_name,
            is_versioned=is_versioned,
            name_length=name_length,
            has_date=has_date,
            relative_path=relative_path
        )

    def analyze_directory(self, dir_path: GSPath) -> bool:
        """Analyze if a directory has version-like naming."""
        return any(re.search(pattern, dir_path.name) for pattern in self.version_patterns)

    def is_bucket_public(self) -> bool:
        """Check if bucket is public."""
        try:
            unauthenticated_client = GSPath(self.bucket_path, anon=True)
            next(unauthenticated_client.iterdir())
            return True
        except Exception:
            return False

    def run_analysis(self) -> Dict:
        """Run complete bucket analysis in a single pass."""
        # Initialize aggregation containers
        extension_counts = defaultdict(int)
        total_size = 0
        file_count = 0
        readme_exists = False
        doc_files = []
        versioned_items = {'files': [], 'directories': []}
        name_lengths = []
        files_with_dates = []
        
        # Process all files in a single pass
        for item in self.bucket_path.rglob('*'):
            if item.is_file():
                # Run file analysis
                analysis = self.analyze_file(item)
                
                # Aggregate results
                file_count += 1
                total_size += analysis.size_bytes
                extension_counts[analysis.extension] += 1
                name_lengths.append(analysis.name_length)
                
                if analysis.is_readme:
                    readme_exists = True
                if analysis.has_doc_in_name:
                    doc_files.append(analysis.relative_path)
                if analysis.is_versioned:
                    versioned_items['files'].append(analysis.relative_path)
                if analysis.has_date:
                    files_with_dates.append(analysis.relative_path)
            
            elif item.is_dir():
                # Check for versioned directories
                if self.analyze_directory(item):
                    versioned_items['directories'].append(
                        str(item.relative_to(self.bucket_path))
                    )

        # Calculate directory count (subtract 1 for the root)
        dir_count = sum(1 for _ in self.bucket_path.rglob('*') if _.is_dir()) - 1
        
        return {
            'bucket_path': str(self.bucket_path),
            'subdirectory_count': dir_count,
            'is_public': self.is_bucket_public(),
            'total_size_bytes': total_size,
            'file_count': file_count,
            'extension_counts': dict(extension_counts),
            'has_readme': readme_exists,
            'doc_files': doc_files,
            'versioned_items': versioned_items,
            'filename_analysis': {
                'length_stats': {
                    'min': min(name_lengths) if name_lengths else 0,
                    'max': max(name_lengths) if name_lengths else 0,
                    'avg': sum(name_lengths) / len(name_lengths) if name_lengths else 0
                },
                'files_with_dates': files_with_dates
            }
        }

def format_size(size_bytes: int) -> str:
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def print_analysis(analysis: Dict):
    """Print analysis results in a readable format."""
    print(f"\nAnalysis for bucket: {analysis['bucket_path']}")
    print("-" * 50)
    print(f"Number of subdirectories: {analysis['subdirectory_count']}")
    print(f"Is public: {analysis['is_public']}")
    print(f"Total size: {format_size(analysis['total_size_bytes'])}")
    print(f"Total files: {analysis['file_count']}")
    
    print("\nFile extensions:")
    for ext, count in analysis['extension_counts'].items():
        print(f"  {ext}: {count}")
    
    print(f"\nREADME file present: {analysis['has_readme']}")
    
    if analysis['doc_files']:
        print("\nFiles with 'doc' in name:")
        for doc_file in analysis['doc_files']:
            print(f"  {doc_file}")
    
    if any(analysis['versioned_items'].values()):
        print("\nVersioned items:")
        for item_type, items in analysis['versioned_items'].items():
            if items:
                print(f"\n  {item_type.capitalize()}:")
                for item in items:
                    print(f"    {item}")
    
    filename_analysis = analysis['filename_analysis']
    print("\nFilename analysis:")
    print(f"  Length - Min: {filename_analysis['length_stats']['min']}, "
          f"Max: {filename_analysis['length_stats']['max']}, "
          f"Avg: {filename_analysis['length_stats']['avg']:.2f}")
    
    if filename_analysis['files_with_dates']:
        print("\nFiles with dates in name:")
        for file in filename_analysis['files_with_dates'][:5]:  # Show first 5
            print(f"  {file}")
        if len(filename_analysis['files_with_dates']) > 5:
            print(f"  ... and {len(filename_analysis['files_with_dates']) - 5} more")


def analyze_bucket(bucket_path: str) -> Dict:
    """Analyze a GCS bucket and return comprehensive results."""
    analyzer = GCSBucketAnalyzer(bucket_path)
    return analyzer.run_analysis()


def main():
    """Main function to analyze multiple buckets."""
    buckets = [
        "gs://your-bucket-1",
        "gs://your-bucket-2"
    ]
    
    for bucket in buckets:
        try:
            analysis = analyze_bucket(bucket)
            print_analysis(analysis)
        except Exception as e:
            print(f"Error analyzing bucket {bucket}: {str(e)}")

if __name__ == "__main__":
    main()