import os
import subprocess
import time
import csv
import re
import shutil
import matplotlib.pyplot as plt

# ========================================
# CONFIGURATION
# ========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CUDA_DIR = os.path.join(BASE_DIR, "cuda")
CUDA_BUILD = os.path.join(CUDA_DIR, "build")

CPU_DIR = os.path.join(BASE_DIR, "cpuversions")
CPU_BUILD = os.path.join(CPU_DIR, "build")

PY_DIR = os.path.join(BASE_DIR, "python")
DATA_DIR = os.path.join(BASE_DIR, "data")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")
RESULTS_DIR = os.path.join(BASE_DIR, "finalresults")

os.makedirs(RESULTS_DIR, exist_ok=True)

PYTHON_SCRIPTS = ["cnn_only.py", "pooling_only.py", "fc_only.py", "pipeline.py"]

CUDA_FILES = {
    "cnn_gpu": "cnn_gpu.cu",
    "pooling_gpu": "pooling_gpu.cu",
    "fc_gpu": "fc_gpu.cu",
    "pipeline_gpu": "pipeline_gpu.cu"
}

CPU_FILES = {
    "cnn_cpu": "cnn_cpu.cpp",
    "pooling_cpu": "pooling_cpu.cpp",
    "fc_cpu": "fc_cpu.cpp",
    "pipeline_cpu": "pipeline_cpu.cpp"
}

# ========================================
# CLEAN BUILD FOLDERS
# ========================================

def clean_build_folders():
    print("\n🧹 Cleaning old build folders...")
    for path in [CUDA_BUILD, CPU_BUILD]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

clean_build_folders()

# ========================================
# HELPERS
# ========================================

def run_python_script(script):
    path = os.path.join(PY_DIR, script)
    print(f"\n🐍 Running Python: {script}")
    start = time.time()
    result = subprocess.run(["python", path], capture_output=True, text=True)
    end = time.time()
    runtime = end - start
    accuracy = extract_accuracy(result.stdout)
    if result.stderr:
        print("⚠️ Python Error:\n", result.stderr)
    print(result.stdout)
    return runtime, accuracy

def compile_file(source_path, output_path, compiler, flags, extra_sources=None):
    try:
        cmd = [compiler] + flags
        if extra_sources:
            cmd += extra_sources
        cmd += ["-o", output_path, source_path]
        print(f"\n🔧 Compiling: {source_path} → {output_path}")
        subprocess.run(cmd, check=True)
        print("✅ Build successful!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Compilation failed for {source_path}")
        print("Command:", " ".join(cmd))
        print("Error:", e)

def run_executable(exec_path):
    print(f"\n🚀 Running: {exec_path}")
    
    # Run from the PROJECT ROOT so all relative paths work correctly
    # Executables will access data/, exports/, and finalresults/ from root
    start = time.time()
    result = subprocess.run([os.path.abspath(exec_path)], capture_output=True, text=True, cwd=BASE_DIR)
    end = time.time()
    runtime = end - start
    stdout = result.stdout
    if result.stderr:
        print("⚠️ Errors/Warnings:\n", result.stderr)
    print(stdout)
    return runtime, extract_accuracy(stdout)

def extract_accuracy(text):
    match = re.search(r"Accuracy[:\s]+([0-9]+\.[0-9]+)%", text)
    return float(match.group(1)) if match else None

# ========================================
# STEP 1: RUN PYTHON TRAINING
# ========================================

print("\n🐍 RUNNING PYTHON PIPELINE")
performance = []

for script in PYTHON_SCRIPTS:
    runtime, accuracy = run_python_script(script)
    performance.append(["Python", script.replace(".py", ""), f"{runtime:.4f}", accuracy or "N/A"])

# ========================================
# STEP 2: COMPILE CUDA + CPU
# ========================================

print("\n🔨 COMPILING CUDA + CPU FILES")

# ---- CUDA Builds ----
for name, file in CUDA_FILES.items():
    compile_file(
        os.path.join(CUDA_DIR, file),
        os.path.join(CUDA_BUILD, name + ".exe"),
        "nvcc", ["-O2", "-arch=sm_61"]
    )

# ---- CPU Builds (fixed linking path) ----
UTILS_PATH = os.path.join(CPU_DIR, "utils_cpu.cpp")

for name, file in CPU_FILES.items():
    src_path = os.path.join(CPU_DIR, file)
    out_path = os.path.join(CPU_BUILD, name + ".exe")

    if not os.path.exists(UTILS_PATH):
        print(f"❌ Missing utils_cpu.cpp file at: {UTILS_PATH}")
        continue

    compile_file(
        src_path,
        out_path,
        "g++",
        ["-O2", "-std=c++17"],
        extra_sources=[UTILS_PATH]
    )

# ========================================
# STEP 3: RUN CUDA + CPU EXECUTABLES
# ========================================

print("\n🚀 RUNNING GPU & CPU MODULES")
print(f"   All executables will run from: {BASE_DIR}")
print(f"   Data directory: {DATA_DIR}")
print(f"   Exports directory: {EXPORTS_DIR}")
print(f"   Results directory: {RESULTS_DIR}")

for category, build_path in [("GPU", CUDA_BUILD), ("CPU", CPU_BUILD)]:
    for name in (CUDA_FILES if category == "GPU" else CPU_FILES):
        exec_path = os.path.join(build_path, name + ".exe")
        if os.path.exists(exec_path):
            runtime, acc = run_executable(exec_path)
            performance.append([category, name, f"{runtime:.4f}", acc or "N/A"])
        else:
            print(f"⚠️ Missing executable: {exec_path}")

# ========================================
# STEP 4: SAVE RESULTS
# ========================================

csv_path = os.path.join(RESULTS_DIR, "performance_comparison.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Category", "Model", "Runtime (s)", "Accuracy (%)"])
    writer.writerows(performance)

print(f"\n📁 Results saved to: {csv_path}")

# ========================================
# STEP 5: PLOT RESULTS
# ========================================

models = sorted(set([x[1].split('_')[0].upper() for x in performance if x[1] != "pipeline"]))
categories = ["Python", "GPU", "CPU"]

runtime_data = {c: [] for c in categories}

for m in models:
    for c in categories:
        row = next((x for x in performance if x[0] == c and m.lower() in x[1]), None)
        runtime_data[c].append(float(row[2]) if row else None)

x = range(len(models))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar([i - width for i in x], runtime_data["Python"], width, label="Python")
plt.bar(x, runtime_data["GPU"], width, label="GPU")
plt.bar([i + width for i in x], runtime_data["CPU"], width, label="CPU")
plt.xticks(x, models)
plt.ylabel("Runtime (seconds)")
plt.title("Performance Comparison")
plt.legend()
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "performance_comparison.png"))
plt.show()

print("\n✅ ALL DONE SUCCESSFULLY ✅")