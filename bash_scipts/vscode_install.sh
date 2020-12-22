# save this script to vscode_install.sh 
# run using-> sh vscode_install.sh
VERSION="3.8.0"
PORT=8080
INSTALL_DIR="/home/ubuntu"
wget https://github.com/cdr/code-server/releases/download/${VERSION}/code-server${VERSION}-linux-arm64.tar.gz
tar -xvzf code-server${VERSION}-linux-arm64.tar.gz
rm code-server${VERSION}-${RELEASE}-linux-arm64.tar.gz
mv code-server${VERSION}-${RELEASE}-linux-arm64/  "${INSTALL_DIR}/vscode-server"
echo "cd \"${INSTALL_DIR}/vscode-server\" \n./code-server --no-auth --port ${PORT} --auth none  --cert & \n echo \"\`curl http://169.254.169.254/latest/meta-data/public-hostname\`:${PORT}\"" > start_vscode.sh
cd ${INSTALL_DIR}/vscode-server
./code-server --no-auth --port ${PORT} --auth none --cert &
sleep 5
# this optional line is to make the current user owner of code-server config and extenstions directory 
sudo chown -R ubuntu:ubuntu /home/ubuntu/.local/share/code-server/*