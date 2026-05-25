# Payment Delay Inference API

FastAPI backend for part 2 of the practical project. The service loads the model artifacts produced in part 1 and exposes them through a prediction endpoint.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python -m uvicorn src.api.main:app --reload
```

Open the interactive docs at:

```text
http://127.0.0.1:8000/docs
```

## Endpoints

- `GET /health` returns API status and whether the model artifacts are loaded.
- `GET /features` returns the expected feature names.
- `POST /predict` returns the predicted `payment_delay` value.

The API expects these files to exist:

- `models/best_model.pkl`
- `models/preprocessor.pkl`
- `models/scaler.pkl`

## Example Request

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body '{
    "state": "HI",
    "account_length": 33,
    "area_code": "area_code_415",
    "international_plan": "no",
    "voice_mail_plan": "no",
    "number_vmail_messages": 0,
    "total_day_minutes": 200.5,
    "total_day_calls": 117,
    "total_day_charge": 34.09,
    "total_eve_minutes": 159.9,
    "total_eve_calls": 111,
    "total_eve_charge": 13.59,
    "total_night_minutes": 196.2,
    "total_night_calls": 84,
    "total_night_charge": 8.83,
    "total_intl_minutes": 16.3,
    "total_intl_calls": 6,
    "total_intl_charge": 4.4,
    "number_customer_service_calls": 3
  }'
```
