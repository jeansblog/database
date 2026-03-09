from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

app = Flask(__name__, template_folder="templates")
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/execute', methods=['POST'])
def execute():
    data = request.json or {}
    db_type = data.get('db_type')
    host = data.get('host')
    port = data.get('port')
    user = data.get('user')
    password = data.get('password')
    database = data.get('database')
    sql = data.get('sql')

    if not sql:
        return jsonify({'error': 'SQLが空です'}), 400

    if db_type == 'postgres':
        port = port or 5432
        url = f'postgresql://{user}:{password}@{host}:{port}/{database}'
    elif db_type == 'mysql':
        port = port or 3306
        url = f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}'
    elif db_type == 'oracle':
        port = port or 1521
        # treat `database` as the Oracle service name
        if database:
            url = f'oracle+oracledb://{user}:{password}@{host}:{port}/?service_name={database}'
        else:
            url = f'oracle+oracledb://{user}:{password}@{host}:{port}'
    else:
        return jsonify({'error': '未対応のDBタイプ'}), 400

    try:
        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            if result.returns_rows:
                rows = [list(r) for r in result.fetchall()]
                cols = list(result.keys())
                return jsonify({'columns': cols, 'rows': rows})
            else:
                return jsonify({'rowcount': result.rowcount})
    except SQLAlchemyError as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)