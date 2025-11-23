app = FastAPI()

@app.post("/call")
async def start_call(req: Request):
  body = await req.json()
  from_no = body.get("from")
  to_no = body.get("to")
  account_sid = os.getenv("TWILIO_ACCOUNT_SID")
  auth_token = os.getenv("TWILIO_AUTH_TOKEN")
  client = Client(account_sid, auth_token)

  # Use the public URL of your application
  # here we're using ngrok to expose an app
  # running locally
  call = client.calls.create(
    to=to_no,
    from_=from_no,
    url="https://[your_ngrok_subdomain].ngrok.app/incoming-call"
  )

  return {"sid": f"{call.sid}"}

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(req: Request):
  from twilio.twiml.voice_response import VoiceResponse, Connect
  response = VoiceResponse()
  response.say("Connecting to AI assistant")
  connect = Connect()
  connect.stream(url=f'wss://{req.url.hostname}/media-stream')
  response.append(connect)
  return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
  # stream is a FastRTC stream defined elsewhere
  await stream.telephone_handler(websocket)

app = gr.mount_gradio_app(app, stream.ui, path="/")