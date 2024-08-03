由于本地mac的python通过homebrew安装，所以是一个外部管理环境，
只能通过每次创建虚拟环境来使用python的相关功能，例如pip3 install

创建步骤：

1. python3 -m venv myenv # 创建虚拟环境
2. source myenv/bin/activate # 激活虚拟环境
3. pip install pandas # 在虚拟环境中安装python包

其他方案：使用pipx来安装包，因为pipx是一个专门用于在独立的虚拟环境中安装和管理 Python 应用的工具。它的优势在于，每个应用都有自己独立的环境，避免了不同应用之间的依赖冲突。

使用步骤：

1. 使用homebrew安装pipx
brew install pipx
pipx ensurepath
2. 安装pandas：pipx install pandas





