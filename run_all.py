import os
import subprocess
import time
import csv
import re
import matplotlib.pyplot as plt

# ========================================
# CONFIGURATION
# ========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUDA_DIR = os.path.join(BASE_DIR, "cuda")
CPU_DIR = os.path.join(BASE_DIR, "cpuversions")
PY_DIR = os.path.join(BASE_DIR, "python")
RESULTS_DIR = os.path.join(BASE_DIR, "finalresults")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Ordered Python pipeline
PYTHON_SCRIPTS = [
    "cnnonly.py",
    "poolonly.py",
    "fconly.py",
    "pipeline.py"
]

# GPU CUDA files
CUDA_FILES = {
    "cnn_gpu": "cnn_gpu.cu",
    "pooling_gpu": "pooling_gpu.cu",
    "fc_gpu": "fc_gpu.cu",
    "pipeline_gpu": "pipeline_gpu.cu"
}

# CPU C++ files
CPU_FILES = {
    "cnn_cpu": "cnn_cpu.cpp",
    "pooling_cpu": "pooling_cpu.cpp",
    "fc_cpu": "fc_cpu.cpp",
    "pipeline_cpu": "pipeline_cpu.cpp"
}

# ========================================
# HELPER FUNCTIONS
# ========================================

def run_python_script(script):
    """Run a Python file and measure its execution time."""
    path = os.path.join(PY_DIR, script)
    print(f"\n🐍 Running Python script: {script}")
    start = time.time()
    result = subprocess.run(["python", path], capture_output=True, text=True)
    end = time.time()
    runtime = end - start

    accuracy = extract_accuracy(result.stdout)
    print(result.stdout)
    if result.stderr:
        print("⚠️ Python error:\n", result.stderr)

    return runtime, accuracy

def compile_file(source_path, output_path, compiler, flags):
    """Compile CUDA or CPU source file."""
    try:
        cmd = [compiler] + flags + ["-o", output_path, source_path]
        print(f"\n🔧 Compiling: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"✅ Compiled successfully: {output_path}")
    except subprocess.CalledProcessError:
        print(f"❌ Compilation failed for {source_path}")

def run_executable(exec_path):
    """Run an executable and extract performance info."""
    print(f"\n🚀 Running: {exec_path}")
    start = time.time()
    result = subprocess.run(exec_path, shell=True, capture_output=True, text=True)
    end = time.time()
    runtime = end - start

    stdout = result.stdout
    if result.stderr:
        print("⚠️ Errors/Warnings:\n", result.stderr)
    print(stdout)

    accuracy = extract_accuracy(stdout)
    return runtime, accuracy

def extract_accuracy(output_text):
    """Try to find an accuracy percentage in text (e.g., 'Accuracy: 89.73%')."""
    match = re.search(r"Accuracy[:\s]+([0-9]+\.[0-9]+)%", output_text)
    return float(match.group(1)) if match else None

# ========================================
# STEP 1: RUN PYTHON FILES FIRST
# ========================================
print("\n======================================")
print("🐍 RUNNING PYTHON PIPELINE (TRAINING & EXPORT)")
print("======================================")

performance_data = []

for script in PYTHON_SCRIPTS:
    runtime, accuracy = run_python_script(script)
    performance_data.append(["Python", script.replace(".py", ""), f"{runtime:.4f}", accuracy or "N/A"])

# ========================================
# STEP 2: COMPILE CUDA + CPU FILES
# ========================================
print("\n======================================")
print("🔨 COMPILING CUDA & CPU FILES")
print("======================================")

for name, file in CUDA_FILES.items():
    src = os.path.join(CUDA_DIR, file)
    out = os.path.join(CUDA_DIR, name + ".exe")
    compile_file(src, out, "nvcc", ["-O2", "-arch=sm_61"])

for name, file in CPU_FILES.items():
    src = os.path.join(CPU_DIR, file)
    out = os.path.join(CPU_DIR, name + ".exe")
    compile_file(src, out, "g++", ["-O2"])

# ========================================
# STEP 3: RUN EXECUTABLES
# ========================================
print("\n======================================")
print("🚀 RUNNING GPU & CPU EXECUTABLES")
print("======================================")

for category, fileset, folder in [("GPU", CUDA_FILES, CUDA_DIR), ("CPU", CPU_FILES, CPU_DIR)]:
    for name, file in fileset.items():
        exec_path = os.path.join(folder, name + ".exe")
        if os.path.exists(exec_path):
            runtime, accuracy = run_executable(exec_path)
            performance_data.append([category, name, f"{runtime:.4f}", accuracy or "N/A"])
        else:
            print(f"⚠️ Missing executable: {exec_path}")

# ========================================
# STEP 4: SAVE RESULTS
# ========================================
csv_path = os.path.join(RESULTS_DIR, "performance_comparison.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Category", "Model", "Runtime (s)", "Accuracy (%)"])
    writer.writerows(performance_data)

print(f"\n📁 Results saved to: {csv_path}")

# ========================================
# STEP 5: VISUALIZATION
# ========================================

# Clean and sort data
models = sorted(set([x[1].split('_')[0].upper() for x in performance_data if x[1] != "pipeline"]))
categories = ["Python", "GPU", "CPU"]

runtime_data = {c: [] for c in categories}
accuracy_data = {c: [] for c in categories}

for m in models:
    for c in categories:
        row = next((x for x in performance_data if x[0] == c and m.lower() in x[1]), None)
        runtime_data[c].append(float(row[2]) if row else None)
        accuracy_data[c].append(float(row[3]) if row and row[3] != "N/A" else None)

# Create side-by-side bar charts
x = range(len(models))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar([i - width for i in x], runtime_data["Python"], width, label="Python")
plt.bar(x, runtime_data["GPU"], width, label="GPU")
plt.bar([i + width for i in x], runtime_data["CPU"], width, label="CPU")
plt.xticks(x, models)
plt.ylabel("Runtime (seconds)")
plt.title("Performance Comparison (Python vs GPU vs CPU)")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

plot_path = os.path.join(RESULTS_DIR, "performance_comparison.png")
plt.savefig(plot_path)
plt.show()

print(f"\n📊 Graph saved at: {plot_path}")
print("\n✅ FULL PIPELINE COMPLETED SUCCESSFULLY.")
