# Smart Tomato Farming Assistant

An English/Tamil Streamlit application for tomato leaf screening and locally stored farm planning and record keeping.

## Features

- Tomato leaf disease screening using the existing `tomato_disease_model.keras` model and `tomato_class_names.json`. The model and class order are preserved.
- Plot profiles, scan history, farmer-entered scan confirmations, and CSV exports.
- Weather forecasts for configured Tamil Nadu locations, obtained from Open-Meteo.
- Conservative irrigation planning: records plot, manually selected growth stage, soil type, last logged irrigation, and optional forecast probability. It does not measure soil moisture, prescribe water quantities, or claim forecast rain was observed at the field.
- Manual crop calendar tasks and reminders with editable dates/status, duplicate prevention, and due/overdue display in the app.
- Soil-test data recording, including report context and optional N/P/K, pH, and organic matter values. It does not diagnose deficiencies or calculate fertilizer rates.
- Expense and harvest-sale records with date filtering, sale revenue calculation (`quantity × price`), recorded totals, net result, tables, and chart. Amounts are recorded in INR; these are actual entries, not projections.
- A bilingual farm-data helper with browser-native speech recognition/synthesis where supported, plus typed fallback. Recognition starts only after a user presses the microphone button and grants any browser permission prompt. Browser recognition may transmit audio to the browser's speech provider and may require internet; speech output depends on available device/browser voices.
- Local settings and explicit local-record deletion.

## Farmer-friendly interface

The home screen presents four large actions for checking a plant, viewing weather, opening irrigation planning, and managing farm tasks. Primary navigation is limited to Home, Check Plant, Weather, My Farm Tasks, and More; the remaining farm tools are grouped under More. On narrow screens the primary destinations use a compact bottom navigation bar, while desktop users keep the sidebar. The language selector stays in the page header, and Home is available in both navigation layouts.

Plant checking keeps the existing model and shows its prediction confidence when available, with a clear reminder that the result is not a guaranteed diagnosis. Specific pesticide directions are not shown because this project has no verified local treatment references. Task reminders are grouped into Today, Coming up, and Completed, and can be completed directly from a task card. Weather cards display forecast values and the forecast timestamp only when returned by Open-Meteo.

Responsive styles and Streamlit app-flow tests are included, but no assisted beginner usability study has been performed. Streamlit's actual rendering can differ by browser and device; test on the intended phones and browsers before broad use.

## Requirements and installation

The pinned direct dependencies in `requirements.txt` match the Python 3.12 environment used for local verification. TensorFlow installations can be platform-specific; validate these pins on the intended host before deployment.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The model, label JSON, and treatment JSON must remain alongside `app.py`. `.env.example` only names the optional `TOMATO_APP_DATA_DIR` setting. It is not automatically loaded by Streamlit. For example, configure it in the PowerShell process before starting the app:

```powershell
$env:TOMATO_APP_DATA_DIR = "D:\private-smart-farm-data"
python -m streamlit run app.py
```

Without that setting, SQLite is stored under `%USERPROFILE%\.smart_tomato_farming\farm_records.sqlite3` on Windows. Keep the directory private and on persistent storage. Do not put database files in a public/static directory.

## Data, offline behavior, and privacy

SQLite records and model inference are local and remain available without internet. Weather lookups use the public Open-Meteo endpoint and require a connection; forecast failure does not prevent local record entry. This application does not queue or synchronize changes and has no cross-device backend. Crop tasks/reminders are shown only while using the application and do not notify farmers when it is closed.

This is a local, single-user prototype. It has no authentication, authorization, multi-user data isolation, or secure admin surface. Do not expose it as a public/shared service or treat a shared SQLite file as a secure multi-user database. A public deployment needs authenticated users, per-user authorization, a service-managed database, secure secret management, and tested backup/restore procedures.

To back up local data, stop the app and copy the complete configured data directory to a private, access-controlled location. Verify a restore using a separate copy before relying on the backup. Avoid sharing exported records containing farm information.

## Agricultural recommendation scope

No locally reviewed crop reference dataset is configured. The app does not generate locally validated crop-stage calendars, fertilizer rates, pesticide doses, or treatment prescriptions. Existing treatment data has no usable source/review metadata; chemical entries remain unverified candidates and should not be treated as instructions. Seek qualified local agricultural advice and follow current product labels and regulations.

## Validation and tests

Run the test suite from the project root:

```powershell
python -m unittest discover -s tests -v
```

The tests cover existing model preprocessing/output-label alignment, image validation, treatment schema handling, Streamlit startup/navigation, SQLite schema upgrades and persistence, task deduplication/edit/status, irrigation logs, soil-test validation, financial filters, and INR sales/expense calculations.

During local verification the model file loaded and produced 10 outputs aligned with the 10 configured labels. Its original training code/provenance is not included, so the preprocessing contract cannot be independently compared with training.

Browser voice controls were exercised in the local Streamlit application. Browser speech synthesis completed an utterance; the microphone permission-denied state displayed its fallback guidance. Simulated English and Tamil recognition events verified transcript delivery to the editable field and Tamil locale selection (`ta-IN`). A successful real microphone capture/transcription was not verified because the test browser denied microphone access; support depends on browser, permissions, secure context, audio hardware, and speech-service availability.

## Deployment and operations

No hosting platform has been selected or deployed. Before any deployment, confirm the pinned packages install on the target OS/Python, configure persistent private storage for `TOMATO_APP_DATA_DIR`, and exercise backup/restore. Streamlit exposes a health endpoint at `/_stcore/health`; configure the host's health probe against the app's actual URL and port. A successful local health check is not evidence of a production deployment.
