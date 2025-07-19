#!/usr/bin/env python3
"""
Create visualization charts for presentation
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_performance_comparison():
    """Create performance comparison chart"""
    # Data
    categories = ['ACD F1', 'ACD+SPC F1']
    original = [82.55, 77.32]
    hotel = [99.02, 84.91]
    restaurant = [90.61, 82.00]
    
    x = np.arange(len(categories))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, original, width, label='Original Paper', alpha=0.8, color='#ff7f0e')
    bars2 = ax.bar(x, hotel, width, label='Hotel Domain', alpha=0.8, color='#2ca02c')
    bars3 = ax.bar(x + width, restaurant, width, label='Restaurant Domain', alpha=0.8, color='#1f77b4')
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontweight='bold', fontsize=10)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison: PyTorch vs Original Paper', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig('results/comparison/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_improvement_chart():
    """Create improvement percentage chart"""
    domains = ['Hotel', 'Restaurant', 'Average']
    acd_improvements = [16.47, 8.06, 12.27]
    acd_spc_improvements = [7.59, 4.68, 6.13]
    
    x = np.arange(len(domains))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, acd_improvements, width, label='ACD F1 Improvement', alpha=0.8, color='#2ca02c')
    bars2 = ax.bar(x + width/2, acd_spc_improvements, width, label='ACD+SPC F1 Improvement', alpha=0.8, color='#1f77b4')
    
    # Add value labels
    def add_improvement_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'+{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontweight='bold', fontsize=11)
    
    add_improvement_labels(bars1)
    add_improvement_labels(bars2)
    
    ax.set_xlabel('Domain', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Improvement Over Original Paper', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 20)
    
    plt.tight_layout()
    plt.savefig('results/comparison/improvement_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_domain_complexity_chart():
    """Create domain complexity vs performance chart"""
    domains = ['Hotel', 'Restaurant']
    aspects = [34, 12]
    acd_f1 = [99.02, 90.61]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Aspect count chart
    bars1 = ax1.bar(domains, aspects, alpha=0.8, color=['#2ca02c', '#1f77b4'])
    ax1.set_xlabel('Domain', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Aspects', fontsize=12, fontweight='bold')
    ax1.set_title('Domain Complexity', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontweight='bold', fontsize=12)
    
    # Performance chart
    bars2 = ax2.bar(domains, acd_f1, alpha=0.8, color=['#2ca02c', '#1f77b4'])
    ax2.set_xlabel('Domain', fontsize=12, fontweight='bold')
    ax2.set_ylabel('ACD F1 Score (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Domain Performance', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(85, 100)
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/comparison/domain_complexity.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_architecture_diagram():
    """Create a simple architecture flow diagram"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Hide axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define boxes and arrows
    boxes = [
        {'text': 'Vietnamese Text Input\n"Khách sạn này rất tốt"', 'pos': (5, 9), 'color': '#e6f3ff'},
        {'text': 'PhoBERT Encoder\n(Vietnamese-specific)', 'pos': (5, 7.5), 'color': '#ccebff'},
        {'text': 'Concatenate Last 4 Layers\n768×4 = 3072 dimensions', 'pos': (5, 6), 'color': '#b3e0ff'},
        {'text': 'Multi-task Dense Head\n3072 → num_aspects × 4', 'pos': (5, 4.5), 'color': '#99d6ff'},
        {'text': 'Output per Aspect\n[Neg, Neu, Pos, None]', 'pos': (5, 3), 'color': '#80ccff'},
    ]
    
    # Draw boxes
    for box in boxes:
        rect = plt.Rectangle((box['pos'][0]-1.5, box['pos'][1]-0.4), 3, 0.8, 
                           facecolor=box['color'], edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(box['pos'][0], box['pos'][1], box['text'], 
               ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    for i in range(len(boxes)-1):
        ax.annotate('', xy=(5, boxes[i+1]['pos'][1]+0.4), 
                   xytext=(5, boxes[i]['pos'][1]-0.4), arrowprops=arrow_props)
    
    # Add side annotations
    ax.text(8.5, 7.5, 'Key Innovation:\nVietnamese-specific\nPre-training', 
           ha='center', va='center', fontsize=9, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    ax.text(1.5, 6, 'Multi-layer\nFeatures:\nRicher\nRepresentation', 
           ha='center', va='center', fontsize=9,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    plt.title('PyTorch ABSA Architecture', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('results/comparison/architecture_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_aspect_performance_heatmap():
    """Create heatmap of aspect performance for Hotel domain"""
    # Load Hotel results
    try:
        with open('results/hotel/latest_results_hotel_multitask.json', 'r') as f:
            hotel_data = json.load(f)
        
        aspects = list(hotel_data['test_metrics']['aspect_f1_scores'].keys())
        scores = list(hotel_data['test_metrics']['aspect_f1_scores'].values())
        
        # Reshape into matrix (approximate grouping)
        n_cols = 3
        n_rows = (len(aspects) + n_cols - 1) // n_cols
        
        # Pad with NaN for incomplete rows
        padded_scores = scores + [np.nan] * (n_rows * n_cols - len(scores))
        padded_aspects = aspects + [''] * (n_rows * n_cols - len(aspects))
        
        matrix = np.array(padded_scores).reshape(n_rows, n_cols)
        aspect_matrix = np.array(padded_aspects).reshape(n_rows, n_cols)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
                   cbar_kws={'label': 'F1 Score'}, ax=ax, 
                   xticklabels=False, yticklabels=False)
        
        # Add aspect labels
        for i in range(n_rows):
            for j in range(n_cols):
                if aspect_matrix[i, j]:
                    ax.text(j+0.5, i+0.7, aspect_matrix[i, j], 
                           ha='center', va='center', fontsize=8, 
                           rotation=45 if len(aspect_matrix[i, j]) > 15 else 0)
        
        plt.title('Hotel Domain: Aspect-wise Performance Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/comparison/hotel_aspect_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Could not create heatmap: {e}")

def main():
    """Create all visualizations"""
    print("Creating presentation visualizations...")
    
    # Ensure output directory exists
    os.makedirs('results/comparison', exist_ok=True)
    
    # Create charts
    create_performance_comparison()
    print("✓ Performance comparison chart created")
    
    create_improvement_chart()
    print("✓ Improvement chart created")
    
    create_domain_complexity_chart()
    print("✓ Domain complexity chart created")
    
    create_architecture_diagram()
    print("✓ Architecture diagram created")
    
    create_aspect_performance_heatmap()
    print("✓ Aspect performance heatmap created")
    
    print(f"\n📊 All visualizations saved to results/comparison/")
    print("Charts available:")
    print("- performance_comparison.png")
    print("- improvement_chart.png")
    print("- domain_complexity.png")
    print("- architecture_diagram.png")
    print("- hotel_aspect_heatmap.png")

if __name__ == '__main__':
    main() 