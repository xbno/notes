#!/bin/bash

# using ubuntu 18.04
# this assumes standard base volume already exists
# anaconda3 mounted on efs
# curl -O https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-ppc64le.sh
# bash Anaconda3-2019.03-Linux-x86_64.sh


# extra apt-gets
sudo apt-get update
sudo apt-get install -y \
    awscli \
    postgresql

# setup dse users, and add personal ssh keys
USERS=("unk")
KEYS=("ssh-rsa key-letters-abcdefg")

# prep docker group, to allow access to all users
sudo groupadd docker

for i in "${!USERS[@]}"; do
    sudo adduser ${USERS[i]}
    echo "${USERS[i]} ALL=(ALL) NOPASSWD: ALL" | sudo tee -a /etc/sudoers > /dev/null
    sudo usermod -g hadoop ${USERS[i]}
    sudo usermod -aG docker ${USERS[i]}
    sudo -u ${USERS[i]} mkdir -p /home/${USERS[i]}/.ssh
    sudo -u ${USERS[i]} chmod 700 /home/${USERS[i]}/.ssh
    sudo -u ${USERS[i]} install -m 600 /home/${USERS[i]}/.ssh/ authorized_keys
    sudo -u ${USERS[i]} echo "${KEYS[i]}" | sudo tee -a /home/${USERS[i]}/.ssh/authorized_keys > /dev/null
done

# add git config (geoff only)
echo "[user]" >> /home/user/.gitconfig
echo "        email = email@email.com" >> /home/user/.gitconfig
echo "        name = First Last" >> /home/user/.gitconfig

# must be done by root
# automatically mounts the attached vol /dev/xvdf to /mnt/gc when restarting/reseizing the instance
sudo su
mkdir /mnt/gc
echo "/dev/xvdf       /mnt/gc   ext4    defaults,nofail        0       2" >> /etc/fstab
exit

# pull id_rsa to clone github, don't need this with the dse users part above
# aws s3 cp s3://teams-ent-sessionm-com/admteam/users/gcounihan/keys/id_rsa ~/.ssh/

# docker
sudo apt-get remove docker docker-engine docker.io containerd runc
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-get install docker-ce
sudo add-apt-repository    "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install docker-ce -y

# change base dir for docker

# setup default conda env sourced to /mnt/gc
cat /mnt/gc/.conda_bashrc_add >> ~/.bashrc
