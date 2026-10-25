from PIL import Image
import imagehash
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def visualize_single_hash(image_path, hash_size=8):
    """
    Comprehensive visualization of a single image's hash.
    Shows original image, resized version, hash matrix, and hash details.
    """
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    # Load the image
    img = Image.open(image_path)
    original_size = img.size
    
    # Compute the average hash
    average_h = imagehash.average_hash(img, hash_size=hash_size)
    hash_matrix = average_h.hash
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Image Hashing Process Visualization\n{Path(image_path).name}', fontsize=16, fontweight='bold')
    
    # 1. Original Image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title(f'Original Image\n{original_size}')
    axes[0, 0].axis('off')
    
    # 2. Resized for hashing (8x8 grayscale)
    img_resized = img.resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    img_gray = img_resized.convert('L')  # Convert to grayscale
    axes[0, 1].imshow(img_gray, cmap='gray')
    axes[0, 1].set_title(f'Resized to {hash_size}x{hash_size}\n(Grayscale)')
    axes[0, 1].axis('off')
    
    # 3. Hash Matrix Visualization
    axes[0, 2].imshow(hash_matrix, cmap='gray_r', interpolation='nearest')
    axes[0, 2].set_title(f'Hash Matrix\n(1=White, 0=Black)')
    axes[0, 2].set_xticks(range(hash_size))
    axes[0, 2].set_yticks(range(hash_size))
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Hash matrix with values (binary)
    for i in range(hash_size):
        for j in range(hash_size):
            text = axes[0, 2].text(j, i, '1' if hash_matrix[i, j] else '0',
                                  ha="center", va="center", color="red", fontsize=8)
    
    # 5. Numerical representation of hash
    axes[1, 0].axis('off')
    hash_info = f"""Hash Information:
    
Hex: {str(average_h)}
    
Binary: {bin(int(str(average_h), 16))[2:].zfill(hash_size*hash_size)}

Shape: {hash_matrix.shape}

Size: {hash_size}x{hash_size} = {hash_size*hash_size} bits

Image Path: {Path(image_path).name}
"""
    axes[1, 0].text(0.1, 0.5, hash_info, fontsize=10, family='monospace',
                   verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 6. Pixel intensity heatmap (average grayscale intensity)
    img_array = np.array(img_gray)
    im = axes[1, 1].imshow(img_array, cmap='hot', interpolation='nearest')
    axes[1, 1].set_title('Pixel Intensity Heatmap\n(Used for hash computation)')
    plt.colorbar(im, ax=axes[1, 1])
    
    # 7. Threshold visualization (how hash is derived)
    mean_intensity = np.mean(img_array)
    threshold_matrix = img_array > mean_intensity
    axes[1, 2].imshow(threshold_matrix, cmap='gray_r', interpolation='nearest')
    axes[1, 2].set_title(f'Threshold Applied\n(Mean: {mean_intensity:.1f})')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return average_h


def visualize_hash_comparison(image_paths, hash_size=8):
    """
    Compare hashes of multiple images.
    Shows how similar images have similar hashes.
    """
    if len(image_paths) == 0:
        print("No images provided")
        return
    
    # Compute hashes for all images
    hashes = []
    images_data = []
    
    for path in image_paths:
        if os.path.exists(path):
            img = Image.open(path)
            h = imagehash.average_hash(img, hash_size=hash_size)
            hashes.append(h)
            images_data.append((Path(path).name, img, h))
    
    if len(images_data) == 0:
        print("No valid images found")
        return
    
    # Create comparison figure
    num_images = len(images_data)
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4*num_images))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Hash Comparison Across Multiple Images', fontsize=16, fontweight='bold')
    
    for idx, (filename, img, h) in enumerate(images_data):
        # Show original image
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(f'{filename}')
        axes[idx, 0].axis('off')
        
        # Show hash matrix
        axes[idx, 1].imshow(h.hash, cmap='gray_r', interpolation='nearest')
        axes[idx, 1].set_title(f'Hash: {str(h)[:16]}...')
        axes[idx, 1].axis('off')
        
        # Show Hamming distance to first image (if not first image)
        if idx == 0:
            axes[idx, 2].axis('off')
            axes[idx, 2].text(0.5, 0.5, 'Reference\nImage', 
                             ha='center', va='center', fontsize=14, fontweight='bold')
        else:
            distance = hashes[0] - h
            axes[idx, 2].axis('off')
            axes[idx, 2].text(0.5, 0.5, f'Hamming Distance\nto 1st: {distance}',
                             ha='center', va='center', fontsize=14, fontweight='bold',
                             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    # Print all hash comparisons
    print("\n" + "="*60)
    print("HASH COMPARISON TABLE")
    print("="*60)
    for i, (path1, _, h1) in enumerate(images_data):
        for j, (path2, _, h2) in enumerate(images_data):
            if i < j:
                distance = h1 - h2
                similarity = (1 - distance / (hash_size * hash_size)) * 100
                print(f"{path1} <-> {path2}")
                print(f"  Hamming Distance: {distance}, Similarity: {similarity:.1f}%\n")


def main():
    """
    Main function to run visualizations.
    Uses captured card images from your workspace.
    """
    base_path = 'Captured_Cards'
    
    # Get list of card images
    card_images = sorted([
        os.path.join(base_path, f) for f in os.listdir(base_path)
        if f.endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    if len(card_images) == 0:
        print("No images found in Captured_Cards folder")
        return
    
    print(f"Found {len(card_images)} card images\n")
    
    # 1. Visualize the first image in detail
    print("=" * 60)
    print("DETAILED HASH VISUALIZATION - FIRST IMAGE")
    print("=" * 60)
    hash1 = visualize_single_hash(card_images[0], hash_size=8)
    
    # 2. Compare multiple images
    if len(card_images) >= 2:
        print("\n" + "=" * 60)
        print("HASH COMPARISON - FIRST 4 IMAGES")
        print("=" * 60)
        images_to_compare = card_images[:4]  # Compare first 4 images
        visualize_hash_comparison(images_to_compare, hash_size=8)
    
    # 3. Interactive hash computation for any image
    print("\n" + "=" * 60)
    print("AVAILABLE IMAGES FOR CUSTOM VISUALIZATION")
    print("=" * 60)
    print(f"Total images: {len(card_images)}")
    print("\nTo visualize a specific image, use:")
    print("  visualize_single_hash('path/to/image.jpg')")
    print("\nTo compare specific images, use:")
    print("  visualize_hash_comparison(['path1.jpg', 'path2.jpg', 'path3.jpg'])")
    print("\nExample:")
    print(f"  visualize_single_hash(r'{card_images[0]}')")
    print(f"  visualize_hash_comparison([r'{card_images[0]}', r'{card_images[1]}', r'{card_images[2]}'])")


def visualize_hash_matching(image_path, best_matching_card_info, hamming_distance, 
                            hash_size=8, phash_obj=None, matched_phash_obj=None):
    """
    Comprehensive visualization of image hashing and matching process.
    Shows the input image, best match, their hashes, and differences.
    
    Args:
        image_path: Path to input image
        best_matching_card_info: Dict with card info including 'image_uri' or 'image'
        hamming_distance: Integer hamming distance between hashes
        hash_size: Size of hash matrix (typically 8 or 16)
        phash_obj: The perceptual hash object of input image
        matched_phash_obj: The perceptual hash object of matched card
    """
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    # Load input image
    input_img = Image.open(image_path)
    
    # Compute perceptual hash if not provided
    if phash_obj is None:
        input_phash = imagehash.phash(input_img)
    else:
        input_phash = phash_obj
    
    input_hash_matrix = input_phash.hash
    
    # Load or fetch the matched card image
    try:
        if 'image_uri' in best_matching_card_info:
            import requests
            from io import BytesIO
            response = requests.get(best_matching_card_info['image_uri'], timeout=5)
            if response.status_code == 200:
                matched_img = Image.open(BytesIO(response.content))
            else:
                matched_img = Image.new('RGB', (100, 100), color='gray')
        elif 'image' in best_matching_card_info:
            matched_img = best_matching_card_info['image']
        else:
            matched_img = Image.new('RGB', (100, 100), color='gray')
    except Exception as e:
        print(f"Could not load matched card image: {e}")
        matched_img = Image.new('RGB', (100, 100), color='gray')
    
    # Compute matched card hash if not provided
    if matched_phash_obj is None:
        matched_phash = imagehash.phash(matched_img)
    else:
        matched_phash = matched_phash_obj
    
    matched_hash_matrix = matched_phash.hash
    
    # Compute difference matrix (XOR of hashes)
    diff_matrix = input_hash_matrix != matched_hash_matrix
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    fig.suptitle('Hash Matching Visualization - Input vs Best Match', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # ==== ROW 1: Images and Basic Hashes ====
    # 1.1 Input Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(input_img)
    ax1.set_title('Input Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 1.2 Input Hash Matrix
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(input_hash_matrix, cmap='gray_r', interpolation='nearest')
    ax2.set_title(f'Input Hash\n{str(input_phash)[:16]}...', fontsize=11, fontweight='bold')
    ax2.set_xticks(range(8))
    ax2.set_yticks(range(8))
    ax2.grid(True, alpha=0.3, color='red', linewidth=0.5)
    for i in range(8):
        for j in range(8):
            ax2.text(j, i, '1' if input_hash_matrix[i, j] else '0',
                    ha="center", va="center", color="red", fontsize=8, fontweight='bold')
    
    # 1.3 Matched Card Image
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(matched_img)
    card_name = best_matching_card_info.get('name', 'Unknown')[:20]
    ax3.set_title(f'Best Match: {card_name}', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # 1.4 Matched Hash Matrix
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(matched_hash_matrix, cmap='gray_r', interpolation='nearest')
    ax4.set_title(f'Matched Hash\n{str(matched_phash)[:16]}...', fontsize=11, fontweight='bold')
    ax4.set_xticks(range(8))
    ax4.set_yticks(range(8))
    ax4.grid(True, alpha=0.3, color='red', linewidth=0.5)
    for i in range(8):
        for j in range(8):
            ax4.text(j, i, '1' if matched_hash_matrix[i, j] else '0',
                    ha="center", va="center", color="red", fontsize=8, fontweight='bold')
    
    # ==== ROW 2: Comparison and Differences ====
    # 2.1 Difference Matrix (XOR - where bits differ)
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(diff_matrix, cmap='RdYlGn_r', interpolation='nearest')
    ax5.set_title(f'Difference Matrix\n(Red=Different, Green=Same)', fontsize=11, fontweight='bold')
    ax5.set_xticks(range(8))
    ax5.set_yticks(range(8))
    ax5.grid(True, alpha=0.3, color='blue', linewidth=0.5)
    for i in range(8):
        for j in range(8):
            ax5.text(j, i, 'D' if diff_matrix[i, j] else '✓',
                    ha="center", va="center", 
                    color='white' if diff_matrix[i, j] else 'green', 
                    fontsize=9, fontweight='bold')
    
    # 2.2 Matching Quality Metrics
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.axis('off')
    total_bits = 64
    matching_bits = total_bits - hamming_distance
    similarity_percent = (matching_bits / total_bits) * 100
    
    metrics_text = f"""MATCHING METRICS

Hamming Distance: {hamming_distance}
Matching Bits: {matching_bits}/{total_bits}
Different Bits: {hamming_distance}

Similarity: {similarity_percent:.1f}%

Quality:
{"✓ EXCELLENT" if similarity_percent >= 90 else "✓ GOOD" if similarity_percent >= 80 else "✓ FAIR" if similarity_percent >= 70 else "✗ POOR"}
(threshold: 80%)
"""
    ax6.text(0.05, 0.95, metrics_text, fontsize=11, family='monospace',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            transform=ax6.transAxes, fontweight='bold')
    
    # 2.3 Card Information
    ax7 = fig.add_subplot(gs[1, 2:4])
    ax7.axis('off')
    
    card_info_text = f"""MATCHED CARD INFORMATION

Name: {best_matching_card_info.get('name', 'N/A')}
Set: {best_matching_card_info.get('set', 'N/A')}
Collector #: {best_matching_card_info.get('collector_number', 'N/A')}
Rarity: {best_matching_card_info.get('rarity', 'N/A')}

Scryfall ID: {best_matching_card_info.get('id', 'N/A')[:12]}...
"""
    ax7.text(0.05, 0.95, card_info_text, fontsize=10, family='monospace',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            transform=ax7.transAxes)
    
    # ==== ROW 3: Bit-by-bit comparison ====
    # 3.1 Input bits as heatmap
    ax8 = fig.add_subplot(gs[2, 0])
    im8 = ax8.imshow(input_hash_matrix.astype(int), cmap='coolwarm', interpolation='nearest', vmin=0, vmax=1)
    ax8.set_title('Input: 1=Bright, 0=Dark', fontsize=10, fontweight='bold')
    ax8.set_xticks([])
    ax8.set_yticks([])
    plt.colorbar(im8, ax=ax8, shrink=0.8)
    
    # 3.2 Matched bits as heatmap
    ax9 = fig.add_subplot(gs[2, 1])
    im9 = ax9.imshow(matched_hash_matrix.astype(int), cmap='coolwarm', interpolation='nearest', vmin=0, vmax=1)
    ax9.set_title('Matched: 1=Bright, 0=Dark', fontsize=10, fontweight='bold')
    ax9.set_xticks([])
    ax9.set_yticks([])
    plt.colorbar(im9, ax=ax9, shrink=0.8)
    
    # 3.3 Flipped difference visualization
    ax10 = fig.add_subplot(gs[2, 2])
    diff_display = diff_matrix.astype(int)
    im10 = ax10.imshow(diff_display, cmap='Reds', interpolation='nearest', vmin=0, vmax=1)
    ax10.set_title('Differences (0=Match, 1=Differ)', fontsize=10, fontweight='bold')
    ax10.set_xticks([])
    ax10.set_yticks([])
    plt.colorbar(im10, ax=ax10, shrink=0.8)
    
    # 3.4 Hash strings comparison
    ax11 = fig.add_subplot(gs[2, 3])
    ax11.axis('off')
    
    input_bin = bin(int(str(input_phash), 16))[2:].zfill(64)
    matched_bin = bin(int(str(matched_phash), 16))[2:].zfill(64)
    
    hash_comparison = f"""HEX REPRESENTATION

Input:   {str(input_phash)}

Matched: {str(matched_phash)}

BINARY (first 32 bits):
Input:   {input_bin[:32]}...
Matched: {matched_bin[:32]}...
"""
    ax11.text(0.05, 0.95, hash_comparison, fontsize=9, family='monospace',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
             transform=ax11.transAxes)
    
    plt.tight_layout()
    return fig


def generate_hash_visualization_for_gradio(image_path, best_matching_card_info, 
                                          hamming_distance, hash_size=8,
                                          phash_obj=None, matched_phash_obj=None):
    """
    Generate hash matching visualization as an image file for Gradio.
    Returns the path to the saved figure.
    """
    import tempfile
    
    fig = visualize_hash_matching(image_path, best_matching_card_info, hamming_distance,
                                 hash_size, phash_obj, matched_phash_obj)
    
    if fig is None:
        return None
    
    # Save to temporary file
    temp_file = os.path.join(tempfile.gettempdir(), 'hash_visualization.png')
    fig.savefig(temp_file, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return temp_file


if __name__ == "__main__":
    main()
