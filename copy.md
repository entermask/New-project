source ~/venvs/omnivoice-api/bin/activate

set -a
source .env
set +a

uvicorn app:app --host 0.0.0.0 --port 8001


source ~/venvs/omnivoice-api/bin/activate
pip install -r requirements.txt

git clone https://github.com/entermask/New-project.git

docker exec -it supabase_db_supabase psql -U postgres -c "ALTER USER postgres WITH PASSWORD 'postgres';"