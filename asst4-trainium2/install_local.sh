#!/bin/bash

# 定义虚拟环境文件夹名称
ENV_DIR="nki_env"

# 确保脚本是通过 source 调用的，否则虚拟环境无法在当前终端生效
if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
    echo "⚠️  调用方式错误！"
    echo "请使用命令: source start_local.sh (或 . start_local.sh) 来启动环境。"
    exit 1
fi

# 检查系统是否具备基础 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 未检测到 Python3，请先在终端执行安装:"
    echo "sudo apt update && sudo apt install python3 python3-venv python3-pip -y"
    return 1
fi

# 1. 检查并自动创建虚拟环境
if [ ! -d "$ENV_DIR" ]; then
    echo "🌱 未检测到本地环境，正在自动创建虚拟环境 ($ENV_DIR)..."
    python3 -m venv $ENV_DIR
    
    echo "🔌 正在激活环境并安装 AWS NKI 模拟器及依赖 (可能需要1-2分钟)..."
    source $ENV_DIR/bin/activate
    
    # 升级 pip 并从 AWS 源安装 Neuron 编译器核心包 (内含 neuronxcc.nki)
    pip install --upgrade pip
    pip install neuronx-cc --extra-index-url=https://pip.repos.neuron.amazonaws.com
    pip install numpy pytest
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    
    echo "✅ 环境依赖初始化成功！"
else
    # 环境已存在，直接激活
    source $ENV_DIR/bin/activate
fi

# 2. 打印环境就绪面板
NKI_VER=$(python -c "import neuronxcc.nki; print(neuronxcc.nki.__version__)" 2>/dev/null)

echo ""
echo "======================================================="
echo " 🚀 CS149 Trainium2 本地模拟环境已就绪 "
echo "======================================================="
echo " • 当前激活环境 : $VIRTUAL_ENV"
echo " • NKI SDK 版本 : ${NKI_VER:-未知 (请检查安装)}"
echo "-------------------------------------------------------"
echo " 💡 常用开发指令:"
echo "   运行本地测试 : python ./part1/run_local_test.py --kernel naive -n 128"
echo "   退出开发环境 : deactivate"
echo "======================================================="
echo ""
