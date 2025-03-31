import os
import subprocess
import argparse

def set_mkl_var_and_run_cbm_setup(setvars_path):
    setup_dir = os.path.dirname(os.path.abspath(__file__)) + "/cbm"
    os.chdir(setup_dir)

    # Run setvars.sh in a subshell and preserve environment variables
    command = f"bash -c 'source {setvars_path} --force && env'"
    env_vars = subprocess.check_output(command, shell=True, executable="/bin/bash", text=True)
    
    # Parse environment variables
    env_dict = dict(line.split("=", 1) for line in env_vars.splitlines() if "=" in line)
    os.environ.update(env_dict)

    # Now run the build command with the updated environment
    subprocess.check_call(["python", "setup.py", "build_ext", "--inplace"], env=os.environ)
    
    subprocess.check_call(["python", "setup_cbm_cusparse.py", "build_ext", "--inplace"], env=os.environ)
    subprocess.check_call(["python", "setup_csr_cusparse.py", "build_ext", "--inplace"], env=os.environ)

def cmake_for_arbok():
    setup_dir = os.path.dirname(os.path.abspath(__file__)) + "/arbok"
    os.chdir(setup_dir)

    cmake_command = "[ -d 'build' ] || cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build/"
    subprocess.check_call(cmake_command, shell=True, executable="/bin/bash")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a Python extension with environment variables sourced from Intel's setvars.sh.")
    parser.add_argument("--setvars_path",
                        type=str, default="/opt/intel/oneapi/setvars.sh",
                        help="The path to the setvars.sh script to source Intel environment variables (default: /opt/intel/oneapi/setvars.sh)")
    args = parser.parse_args()
    
    cmake_for_arbok()
    set_mkl_var_and_run_cbm_setup(args.setvars_path)