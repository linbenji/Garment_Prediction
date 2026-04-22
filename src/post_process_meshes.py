import os
import numpy as np
import open3d as o3d
import time

# ==========================================
# CONFIGURATION
# ==========================================
DIRECTORY_PATH = r"C:\Users\chung\Desktop\Garment_Prediction\results\model_v3_master_bend\eval_results_test\meshes"
TEMPLATE_MESH_PATH = r"C:\Users\chung\Desktop\Garment_Prediction\dataset\batch_1500_lean\template_sizes\template.obj"

# Smoothing parameters - Cautious approach to preserve folds
# Simple Laplacian smoothing (keep parameters low to avoid smoothing out folds)
NUM_ITERATIONS = 3   # Increase for more smoothing (and potentially more fold dulling)
LAMBDA_FILTER = 0.1  # Weighted average of neighbors - increase for more movement per iteration (less feature preservation)

# Welding Parameters
WELD_DECIMALS = 4    # Precision for finding overlapping vertices (4 is usually perfect)

# ==========================================
# FUNCTIONS
# ==========================================

def weld_mesh_seams(predicted_verts, template_verts, decimals=WELD_DECIMALS):
    """
    Identifies duplicated seam vertices in the template mesh and forces their 
    predicted positions to be perfectly averaged, zipping the mesh back together.
    """
    # Round template to group vertices that share the exact same starting position
    rounded_template = np.round(template_verts, decimals=decimals)
    
    # Map every vertex to a unique group ID based on its template position
    unique_coords, inverse_indices = np.unique(rounded_template, axis=0, return_inverse=True)
    
    # Create arrays to sum the predicted positions and count duplicates
    sum_verts = np.zeros_like(unique_coords, dtype=np.float64)
    counts = np.zeros(len(unique_coords), dtype=np.float64)
    
    # Accumulate predictions for vertices that share the same template coordinate
    np.add.at(sum_verts, inverse_indices, predicted_verts)
    np.add.at(counts, inverse_indices, 1.0)
    
    # Calculate the perfect average for the seams
    averaged_unique_verts = sum_verts / counts[:, None]
    
    # Map the averaged coordinates back to the full vertex array
    welded_verts = averaged_unique_verts[inverse_indices]
    
    return welded_verts


def post_process_mesh(input_path, output_path, template_verts, iterations, lambda_val):
    """
    Loads an obj mesh, applies Laplacian smoothing, welds seams, and saves the result.
    Args:
        input_path: Full path to input obj mesh
        output_path: Full path to save smoothed obj mesh
        template_verts: Vertices in the template obj file
        iterations: Number of Laplacian smoothing steps
        lambda_val: Smoothing weight per step
    """
    try:
        # 1. Load the mesh using Open3D
        mesh = o3d.io.read_triangle_mesh(input_path)
        if not mesh.has_triangles():
            print(f"Error loading mesh from {input_path} or mesh has no triangles.")
            return
        
        pred_verts = np.asarray(mesh.vertices)
        if len(pred_verts) != len(template_verts):
            print(f"  [Error] Vertex count mismatch in {os.path.basename(input_path)}. "
                  f"Expected {len(template_verts)}, got {len(pred_verts)}.")
            return False

        # 2. Compute vertex normals (required for some smoothing algorithms/visuals)
        mesh.compute_vertex_normals()

        # 3. Apply standard Laplacian smoothing cautiously
        # This will smooth out high-frequency jaggedness but is low-parameter
        # enough that it shouldn't significantly flatten broad folds/wrinkles
        smoothed_mesh = mesh.filter_smooth_laplacian(number_of_iterations=iterations, lambda_filter=lambda_val)

        # 4. Zip the Seams
        smoothed_verts = np.asarray(smoothed_mesh.vertices)
        welded_verts = weld_mesh_seams(smoothed_verts, template_verts, WELD_DECIMALS)

        # Apply the welded vertices back to the Open3D mesh object
        smoothed_mesh.vertices = o3d.utility.Vector3dVector(welded_verts)

        # 5. Compute normals again for the smoothed mesh
        smoothed_mesh.compute_vertex_normals()

        # 6. Save the modified mesh to the new output path
        o3d.io.write_triangle_mesh(output_path, smoothed_mesh)
        return True

    except Exception as e:
        print(f"An error occurred while processing {os.path.basename(input_path)}: {e}")
        return False

def process_directory(directory, template_path, iterations, lambda_val):
    """
    Iterates through a directory, identifying _pred.obj meshes, smoothes them, and welds the seams
    Args:
        directory: Full path to obj directory
        template_path: Filepath to the template obj file
        iterations: Smoothing iterations
        lambda_val: Smoothing lambda parameter
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'. Please adjust DIRECTORY_PATH.")
        return
    if not os.path.isfile(template_path):
        print(f"Error: Template mesh not found at '{template_path}'")
        return
    
    print(f"Loading reference template from: {template_path}")
    template_mesh = o3d.io.read_triangle_mesh(template_path)
    template_verts = np.asarray(template_mesh.vertices)
    print(f"Template loaded with {len(template_verts)} vertices.")

    print(f"\nProcessing directory: {directory}\n{'-'*50}\n")
    start_time = time.time()
    processed_count = 0

    # Go through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is an obj and ends with '_pred.obj'
        if filename.endswith('_pred.obj') and not filename.endswith('_gt.obj'): # explicit safety checks
            input_path = os.path.join(directory, filename)

            # Construct output filename by inserting '_processed' before '.obj'
            base_name, _ = os.path.splitext(filename)
            output_filename = f"{base_name}_processed.obj"
            output_path = os.path.join(directory, output_filename)

            # Smooth and save, leaving _gt.obj meshes untouched
            if post_process_mesh(input_path, output_path, template_verts, iterations, lambda_val):
                processed_count += 1
                print(f"  Processed -> {output_filename}")

    elapsed_time = time.time() - start_time
    print(f"\n{'-'*50}\nFinished processing {processed_count} mesh(es) in {elapsed_time:.2f} seconds.")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    process_directory(DIRECTORY_PATH, TEMPLATE_MESH_PATH, NUM_ITERATIONS, LAMBDA_FILTER)