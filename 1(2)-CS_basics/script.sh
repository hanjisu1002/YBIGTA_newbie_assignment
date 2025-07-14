#!/bin/bash
# 필수 패키지 설치
apt update && apt install -y wget curl git build-essential python3 python3-pip

# anaconda(또는 miniconda)가 존재하지 않을 경우 설치해주세요!
## TODO
# 이미 설치되어 있는지 확인
if ! command -v conda &> /dev/null; then
    echo "[INFO] Miniconda 설치 중..."
    curl -s https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh | bash -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
else
    echo "[INFO] 이미 conda가 설치되어 있습니다."
fi



# Conda 환셩 생성 및 활성화
## TODO
# myenv 환경이 없으면 생성
if ! conda info --envs | grep -q 'myenv'; then
    conda create -y -n myenv python=3.10
fi
# 환경 활성화
source $HOME/miniconda/etc/profile.d/conda.sh
conda activate myenv

## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "[INFO] 가상환경 활성화: 성공"
else
    echo "[INFO] 가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
## TODO
pip install mypy

# Submission 폴더 파일 실행
# 현재 디렉토리: 1(2)-CS_basics
submission_dir="../1(1)-Python/submission"
input_dir="./input"
output_dir="./output"

mkdir -p "$output_dir"

for filepath in "$submission_dir"/*.py; do
    filename=$(basename "$filepath")              # 예: 1_1260.py
    filename_no_prefix="${filename#*_}"           # 예: 1260.py → 1260
    filename_no_ext="${filename_no_prefix%.py}"   # 예: 1260

    input_file="$input_dir/${filename_no_ext}_input"
    output_file="$output_dir/${filename_no_ext}_output"

    if [ -f "$input_file" ]; then
        echo "[INFO] 실행 중: $filename"
        python "$filepath" < "$input_file" > "$output_file"
    else
        echo "[WARNING] $input_file 없음 — 건너뜀"
    fi
done




# mypy 테스트 실행 및 mypy_log.txt 저장
## TODO
mypy "$submission_dir" > "./mypy_log.txt"

# conda.yml 파일 생성
## TODO
conda env export > "./conda.yml"


# 가상환경 비활성화
## TODO
conda deactivate
