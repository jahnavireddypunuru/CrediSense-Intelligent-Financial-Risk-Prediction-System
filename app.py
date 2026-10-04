"""CrediSense - Intelligent Financial Risk Prediction System (FastAPI app)."""
import csv
import io
import json
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from werkzeug.security import check_password_hash, generate_password_hash

from ml.features import CATEGORICAL, LABELS, NUMERIC, VALUE_LABELS, Engineer  # noqa: F401 (Engineer needed to unpickle)

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "instance" / "credisense.db"

app = FastAPI(title="CrediSense", description="Intelligent Financial Risk Prediction System")
app.add_middleware(SessionMiddleware, secret_key="change-me-in-production-credisense", max_age=60 * 60 * 8)
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")


def url_for(name, **kw):
    if name == "static":
        return "/static/" + kw["filename"]
    return str(app.url_path_for(name, **kw))


templates.env.globals["url_for"] = url_for


def flash(request, message, category="success"):
    request.session.setdefault("_flash", []).append([category, message])


def render(request, name, **ctx):
    msgs = request.session.pop("_flash", [])
    ctx.update(
        request=request, user_name=request.session.get("name"), best_model=META["best"],
        ep=request.scope["endpoint"].__name__ if request.scope.get("endpoint") else "",
        get_flashed_messages=lambda with_categories=False: msgs,
    )
    return templates.TemplateResponse(request, name, ctx)


def redirect(path):
    return RedirectResponse(path, status_code=303)


# --------------------------------------------------------------------------- model
META = json.loads((BASE / "models" / "metrics.json").read_text())
MODEL = joblib.load(BASE / "models" / "best_model.joblib")
THRESHOLD = META["threshold"]
REF = META["reference"]

FIELD_LABELS = LABELS
FIELDS = NUMERIC + list(CATEGORICAL)


def parse_form(form):
    row = {}
    for f in NUMERIC:
        row[f] = float(form[f])
    for f, levels in CATEGORICAL.items():
        if form[f] not in levels:
            raise ValueError(f"Invalid {f}")
        row[f] = form[f]
    if not (18 <= row["person_age"] <= 80):
        raise ValueError("Age must be between 18 and 80")
    if row["person_income"] <= 0 or row["loan_amnt"] <= 0:
        raise ValueError("Income and loan amount must be positive")
    if not (0 <= row["person_emp_length"] <= min(50, row["person_age"] - 14)):
        raise ValueError("Employment length is not possible for this age")
    if not (0 < row["loan_int_rate"] <= 40):
        raise ValueError("Interest rate must be between 0 and 40%")
    if row["cb_person_cred_hist_length"] > row["person_age"] - 16:
        raise ValueError("Credit history cannot be longer than the applicant's adult life")
    return row


def risk_band(p):
    if p < 0.20:
        return "Low"
    if p < THRESHOLD:
        return "Moderate"
    return "High"


def predict(row):
    df = pd.DataFrame([row])
    p = float(MODEL.predict_proba(df)[:, 1][0])
    return min(max(p, 0.005), 0.995)  # avoid showing a misleading 0% / 100%


def explain(row, p):
    """Model-agnostic what-if attribution: reset each factor to the portfolio
    reference value and measure how much the default probability changes."""
    out = []
    for f in FIELDS:
        alt = dict(row)
        alt[f] = REF[f]
        d = p - predict(alt)  # positive => this factor pushes risk up
        out.append({"feature": FIELD_LABELS[f], "impact": round(d * 100, 2)})
    out.sort(key=lambda x: -abs(x["impact"]))
    return out[:6]


def recommendation(row, p, band):
    lti = row["loan_amnt"] / row["person_income"]
    tips = []
    if band == "Low":
        tips.append("Applicant profile is strong. Eligible for standard processing and the best-rate tier.")
    if lti > 0.35:
        tips.append(f"Loan is {lti:.0%} of annual income. Consider a smaller amount or a co-applicant.")
    if row["cb_person_default_on_file"] == "Y":
        tips.append("A past default is on file. Manual underwriter review is recommended.")
    if row["loan_grade"] in ("D", "E", "F", "G"):
        tips.append(f"Loan grade {row['loan_grade']} signals weak credit quality. Ask for collateral or a guarantor.")
    if row["loan_int_rate"] > 15:
        tips.append("Interest rate above 15% raises repayment strain. Review pricing against the risk.")
    if row["person_home_ownership"] == "RENT" and lti > 0.25:
        tips.append("Renter with a sizeable loan. Verify income stability and rent obligations.")
    if band == "Moderate":
        tips.append("Borderline case. Approve with tighter terms or verify income documents.")
    if band == "High" and not tips:
        tips.append("Elevated risk. Escalate to a credit officer for manual review.")
    return tips, lti


# --------------------------------------------------------------------------- db
def get_db():
    if not DB_PATH.exists():
        init_db()  # recreate tables if the database file was removed
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL, password TEXT NOT NULL, created TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS assessments(
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
        applicant TEXT NOT NULL, inputs TEXT NOT NULL, probability REAL NOT NULL,
        band TEXT NOT NULL, created TEXT NOT NULL);
    """)
    c.commit()
    c.close()


init_db()


class NeedLogin(Exception):
    pass


@app.exception_handler(NeedLogin)
async def need_login(request: Request, exc: NeedLogin):
    return redirect("/login?next=" + quote(request.url.path))


def current_user(request: Request):
    if "uid" not in request.session:
        raise NeedLogin()
    return request.session["uid"]


# --------------------------------------------------------------------------- auth
@app.get("/", name="home")
def home(request: Request):
    return redirect("/dashboard" if "uid" in request.session else "/login")


@app.get("/register", name="register")
def register_page(request: Request):
    return render(request, "register.html")


@app.post("/register")
async def register_post(request: Request, conn=Depends(get_db)):
    f = await request.form()
    name, email, pw = f.get("name", "").strip(), f.get("email", "").strip().lower(), f.get("password", "")
    if len(name) < 2 or "@" not in email:
        flash(request, "Enter a valid name and email.", "error")
    elif len(pw) < 6:
        flash(request, "Password must be at least 6 characters.", "error")
    elif pw != f.get("confirm"):
        flash(request, "Passwords do not match.", "error")
    else:
        try:
            conn.execute("INSERT INTO users(name,email,password,created) VALUES(?,?,?,?)",
                         (name, email, generate_password_hash(pw), datetime.now().isoformat()))
            conn.commit()
            flash(request, "Account created. Please sign in.")
            return redirect("/login")
        except sqlite3.IntegrityError:
            flash(request, "That email is already registered.", "error")
    return redirect("/register")


@app.get("/login", name="login")
def login_page(request: Request):
    return render(request, "login.html")


@app.post("/login")
async def login_post(request: Request, conn=Depends(get_db)):
    f = await request.form()
    u = conn.execute("SELECT * FROM users WHERE email=?", (f.get("email", "").strip().lower(),)).fetchone()
    if u and check_password_hash(u["password"], f.get("password", "")):
        request.session.clear()
        request.session.update(uid=u["id"], name=u["name"])
        nxt = request.query_params.get("next", "")
        return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else "/dashboard")
    flash(request, "Invalid email or password.", "error")
    return redirect("/login")


@app.get("/logout", name="logout")
def logout(request: Request):
    request.session.clear()
    flash(request, "You have been signed out.")
    return redirect("/login")


# --------------------------------------------------------------------------- pages
@app.get("/dashboard", name="dashboard")
def dashboard(request: Request, uid=Depends(current_user), conn=Depends(get_db)):
    rows = conn.execute("SELECT * FROM assessments WHERE user_id=? ORDER BY id DESC", (uid,)).fetchall()
    total = len(rows)
    counts = {b: sum(r["band"] == b for r in rows) for b in ("Low", "Moderate", "High")}
    avg = sum(r["probability"] for r in rows) / total if total else 0
    volume = sum(json.loads(r["inputs"])["loan_amnt"] for r in rows)
    trend = [{"label": r["created"][5:10], "p": round(r["probability"] * 100, 1)} for r in rows[:12][::-1]]
    return render(request, "dashboard.html", total=total, counts=counts, avg=avg, volume=volume,
                  recent=rows[:6], trend=trend,
                  models=[{"name": m["name"], "auc": round(m["roc_auc"], 3)} for m in META["results"]])


@app.get("/assess", name="assess")
def assess_page(request: Request, uid=Depends(current_user)):
    return render(request, "assess.html", cat=CATEGORICAL, vl=VALUE_LABELS, form={})


@app.post("/assess")
async def assess_post(request: Request, uid=Depends(current_user), conn=Depends(get_db)):
    form = await request.form()
    try:
        row = parse_form(form)
    except (ValueError, KeyError) as e:
        msg = str(e) if isinstance(e, ValueError) and "could not convert" not in str(e) else \
            "Please fill every field with valid numbers."
        flash(request, msg, "error")
        return render(request, "assess.html", cat=CATEGORICAL, vl=VALUE_LABELS, form=dict(form))
    applicant = form.get("applicant", "").strip() or "Unnamed applicant"
    p = predict(row)
    band = risk_band(p)
    cur = conn.execute(
        "INSERT INTO assessments(user_id,applicant,inputs,probability,band,created) VALUES(?,?,?,?,?,?)",
        (uid, applicant, json.dumps(row), p, band, datetime.now().strftime("%Y-%m-%d %H:%M")))
    conn.commit()
    return redirect(f"/result/{cur.lastrowid}")


@app.get("/result/{aid}", name="result")
def result(aid: int, request: Request, uid=Depends(current_user), conn=Depends(get_db)):
    r = conn.execute("SELECT * FROM assessments WHERE id=? AND user_id=?", (aid, uid)).fetchone()
    if not r:
        return redirect("/history")
    row = json.loads(r["inputs"])
    p = r["probability"]
    tips, lti = recommendation(row, p, r["band"])
    return render(request, "result.html", a=r, row=row, p=p, factors=explain(row, p), tips=tips,
                  lti=lti, threshold=THRESHOLD, labels=FIELD_LABELS, vlabels=VALUE_LABELS)


@app.get("/history", name="history")
def history(request: Request, uid=Depends(current_user), conn=Depends(get_db)):
    q = request.query_params.get("q", "").strip()
    band = request.query_params.get("band", "")
    sql, args = "SELECT * FROM assessments WHERE user_id=?", [uid]
    if q:
        sql += " AND applicant LIKE ?"
        args.append(f"%{q}%")
    if band in ("Low", "Moderate", "High"):
        sql += " AND band=?"
        args.append(band)
    rows = conn.execute(sql + " ORDER BY id DESC", args).fetchall()
    items = [dict(r, loan=json.loads(r["inputs"])["loan_amnt"]) for r in rows]
    return render(request, "history.html", items=items, q=q, band=band)


@app.post("/history/{aid}/delete", name="delete")
def delete(aid: int, request: Request, uid=Depends(current_user), conn=Depends(get_db)):
    conn.execute("DELETE FROM assessments WHERE id=? AND user_id=?", (aid, uid))
    conn.commit()
    flash(request, "Assessment deleted.")
    return redirect("/history")


@app.get("/history/export.csv", name="export")
def export(uid=Depends(current_user), conn=Depends(get_db)):
    rows = conn.execute("SELECT * FROM assessments WHERE user_id=? ORDER BY id", (uid,)).fetchall()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "date", "applicant", "default_probability", "risk_band"] + FIELDS)
    for r in rows:
        i = json.loads(r["inputs"])
        w.writerow([r["id"], r["created"], r["applicant"], round(r["probability"], 4), r["band"]]
                   + [i[f] for f in FIELDS])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=credisense_assessments.csv"})


@app.get("/models", name="models")
def models(request: Request, uid=Depends(current_user)):
    return render(request, "models.html", meta=META, labels=LABELS)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
