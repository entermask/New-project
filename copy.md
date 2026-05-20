source ~/venvs/omnivoice-api/bin/activate

set -a
source .env
set +a

uvicorn app:app --host 0.0.0.0 --port 3006


source ~/venvs/omnivoice-api/bin/activate
pip install -r requirements.txt

git clone https://github.com/entermask/New-project.git && cd ./New-project && ./scripts/install.sh


git pull && RESTART=1 ./scripts/run_tmux.sh

docker exec -it supabase_db_supabase psql -U postgres -c "ALTER USER postgres WITH PASSWORD 'postgres';"