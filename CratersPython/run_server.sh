sudo killall ncat
sudo ncat -l -k -v -p 80 -e "`which nc` localhost 8888" &
/home/ieee8023/.local/bin/jupyter-notebook --port=8888
sudo killall ncat
