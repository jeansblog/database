from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import datetime
import decimal

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
    # sanitize SQL early: remove trailing semicolon and surrounding whitespace to avoid DB-specific syntax errors
    sanitized_sql = (sql or '').strip().rstrip(';').strip()
    variables = data.get('variables') or []

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
            # If variables were detected on client side, avoid loading all rows: return total count and only the first N rows
            if variables:
                SAMPLE_LIMIT = 10
                # use sanitized_sql prepared above

                # get total count if possible
                total = None
                try:
                    count_sql = f"SELECT COUNT(*) AS cnt FROM ({sanitized_sql}) sub"
                    cnt_res = conn.execute(text(count_sql))
                    total = int(cnt_res.scalar() or 0)
                except Exception:
                    total = None

                # build sample SQL depending on DB; ensure subquery has an alias (Oracle requires it)
                if db_type == 'oracle':
                    sample_sql = f"SELECT * FROM ({sanitized_sql}) sub WHERE ROWNUM <= {SAMPLE_LIMIT}"
                else:
                    # Postgres/MySQL: wrap and use LIMIT
                    sample_sql = f"SELECT * FROM ({sanitized_sql}) sub LIMIT {SAMPLE_LIMIT}"

                sample_res = conn.execute(text(sample_sql))
                raw_rows = sample_res.fetchall()
                cols = list(sample_res.keys())

                def serialize_value(v):
                    if v is None:
                        return None
                    if isinstance(v, bytes):
                        return v.decode('utf-8', errors='replace')
                    if isinstance(v, (datetime.date, datetime.datetime, datetime.time)):
                        return v.isoformat()
                    if isinstance(v, decimal.Decimal):
                        try:
                            return float(v)
                        except Exception:
                            return str(v)
                    return v

                rows = [[serialize_value(cell) for cell in row] for row in raw_rows]

                lower_cols = [str(c).lower() for c in cols]
                var_map = {}
                for v in variables:
                    try:
                        var_map[v] = lower_cols.index(v.lower())
                    except ValueError:
                        var_map[v] = -1

                return jsonify({'columns': cols, 'rows': rows, 'variables': variables, 'variable_columns': var_map, 'total_count': total, 'truncated': True})
            else:
                # execute sanitized SQL to avoid trailing-semicolon errors
                result = conn.execute(text(sanitized_sql))
                if result.returns_rows:
                    raw_rows = result.fetchall()
                    cols = list(result.keys())

                    def serialize_value(v):
                        if v is None:
                            return None
                        if isinstance(v, bytes):
                            return v.decode('utf-8', errors='replace')
                        if isinstance(v, (datetime.date, datetime.datetime, datetime.time)):
                            return v.isoformat()
                        if isinstance(v, decimal.Decimal):
                            try:
                                return float(v)
                            except Exception:
                                return str(v)
                        return v

                    rows = [[serialize_value(cell) for cell in row] for row in raw_rows]
                    return jsonify({'columns': cols, 'rows': rows})
    except SQLAlchemyError as e:
        # include sanitized SQL in error response to aid debugging (do not include credentials)
        try:
            info = {'error': str(e), 'executed_sql': sanitized_sql}
        except Exception:
            info = {'error': str(e)}
        return jsonify(info), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)