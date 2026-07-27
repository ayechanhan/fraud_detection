# Deploying to a small EC2 instance (the simple way)

The goal: run the exact same app on an EC2 instance so it's reachable over the
internet.

## 1. Launch the instance

- EC2 → **Launch instance** → **Ubuntu Server 24.04**, type **t3.micro** (free-tier eligible).
- Create + download a key pair (`.pem`) for SSH.
- **Security group** (the firewall) — allow inbound:
  - **SSH (port 22)** from _My IP_
  - **Custom TCP (port 8000)** from _Anywhere_ — the API port (it's protected by the API key)
  - leave everything else closed

## 2. Connect

```bash
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

## 3. One-time setup

```bash
sudo apt update
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev liblzma-dev software-properties-common
sudo add-apt-repository ppa:deadsnakes/pap
sudo apt update && sudo apt install -y python3.13 python3.13-venv python3.13-dev tmux

git clone https://github.com/ayechanhan/fraud_detection.git
cd fraud_detection

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate sample data (no Kaggle download), build the batches, train the model
python automation/make_sample_data.py --out data/raw/paysim.csv
python data/prepare_monthly_batches.py
python model/train.py

# Set the API key
cp .env.example .env
nano .env                      # set FRAUD_API_KEY to a secret of your choice
```

## 4. Run the API

Run it inside a `tmux` session so it stays up after you disconnect:

```bash
tmux new -s api
source .venv/bin/activate
python service/app.py          # serves on 0.0.0.0:8000
# press Ctrl-b then d to detach — the app keeps running in the background
```

## 5. Testing

```bash
curl http://<EC2_PUBLIC_IP>:8000/health

curl -X POST http://<EC2_PUBLIC_IP>:8000/predict \
  -H "X-API-Key: <the key you set>" \
  -H "Content-Type: application/json" \
  -d '{"type":"TRANSFER","amount":500000,"oldbalanceOrg":500000,"newbalanceOrig":0,"oldbalanceDest":0,"newbalanceDest":0}'
```

## 6. View the mlflow UI (optional — keep it private)

```bash
ssh -i your-key.pem -L 5000:127.0.0.1:5000 ubuntu@<EC2_PUBLIC_IP>
# then, on the instance:
cd ~/fraud_detection && source .venv/bin/activate
mlflow ui --backend-store-uri sqlite:///mlflow.db
# now open http://localhost:5000 in browser
```

## 7. Update the deployment later

```bash
cd ~/fraud_detection && git pull
tmux attach -t api             # Ctrl-c to stop the running app, then restart it:
python service/app.py
```
