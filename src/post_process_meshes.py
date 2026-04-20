import os
import open3d as o3d
import time

# ==========================================
# CONFIGURATION
# ==========================================
DIRECTORY_PATH = r"C:\Users\chung\Desktop\Garment_Prediction\results\model_v3_master_bend\eval_results_test\meshes"

# Smoothing parameters - Cautious approach to preserve folds
# Simple Laplacian smoothing (keep parameters low to avoid smoothing out folds)
NUM_ITERATIONS = 3   # Increase for more smoothing (and potentially more fold dulling)
LAMBDA_FILTER = 0.1  # Weighted average of neighbors - increase for more movement per iteration (less feature preservation)

# ==========================================
# FUNCTIONS
# ==========================================

def smooth_pred_mesh(input_path, output_path, iterations, lambda_val):
    """
    Loads an obj mesh, applies Laplacian smoothing, and saves the result.
    Args:
        input_path: Full path to input obj mesh
        output_path: Full path to save smoothed obj mesh
        iterations: Number of Laplacian smoothing steps
        lambda_val: Smoothing weight per step
    """
    try:
        # 1. Load the mesh using Open3D
        mesh = o3d.io.read_triangle_mesh(input_path)
        if not mesh.has_triangles():
            print(f"Error loading mesh from {input_path} or mesh has no triangles.")
            return

        # 2. Compute vertex normals (required for some smoothing algorithms/visuals)
        mesh.compute_vertex_normals()

        # 3. Apply standard Laplacian smoothing cautiously
        # This will smooth out high-frequency jaggedness but is low-parameter
        # enough that it shouldn't significantly flatten broad folds/wrinkles.
        # Open3D's filter_smooth_laplacian is a standard choice.
        # You can increase iterations if more smoothing is needed,
        # but be careful with fold degradation.
        smoothed_mesh = mesh.filter_smooth_laplacian(number_of_iterations=iterations, lambda_filter=lambda_val)

        # 4. Compute normals again for the smoothed mesh
        smoothed_mesh.compute_vertex_normals()

        # 5. Save the modified mesh to the new output path
        o3d.io.write_triangle_mesh(output_path, smoothed_mesh)
        print(f"Successfully smoothed and saved: {os.path.basename(output_path)}")

    except Exception as e:
        print(f"An error occurred while processing {os.path.basename(input_path)}: {e}")

def process_directory(directory, iterations, lambda_val):
    """
    Iterates through a directory, identifying _pred.obj meshes and smoothing them.
    Args:
        directory: Full path to obj directory
        iterations: Smoothing iterations
        lambda_val: Smoothing lambda parameter
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'. Please adjust DIRECTORY_PATH.")
        return

    print(f"\nProcessing directory: {directory}\n{'-'*50}\n")
    start_time = time.time()
    processed_count = 0

    # Go through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is an obj and ends with '_pred.obj'
        if filename.endswith('_pred.obj') and not filename.endswith('_gt.obj'): # explicit safety checks
            input_path = os.path.join(directory, filename)

            # Construct output filename by inserting '_smoothed' before '.obj'
            base_name, _ = os.path.splitext(filename)
            output_filename = f"{base_name}_smoothed.obj"
            output_path = os.path.join(directory, output_filename)

            # Smooth and save, leaving _gt.obj meshes untouched
            smooth_pred_mesh(input_path, output_path, iterations, lambda_val)
            processed_count += 1

    elapsed_time = time.time() - start_time
    print(f"\n{'-'*50}\nFinished processing {processed_count} mesh(es) in {elapsed_time:.2f} seconds.")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    process_directory(DIRECTORY_PATH, NUM_ITERATIONS, LAMBDA_FILTER)