"""Browser-native speech recognition and speech synthesis for the Streamlit UI."""

import streamlit as st


_VOICE_WIDGET = st.components.v2.component(
    "farm_ai_browser_voice",
    html="""
<section class="farm-voice" aria-label="Browser voice controls">
  <div class="farm-voice__controls">
    <button class="farm-voice__button farm-voice__button--primary" data-action="start" type="button"></button>
    <button class="farm-voice__button" data-action="stop" type="button"></button>
    <button class="farm-voice__button" data-action="speak" type="button"></button>
  </div>
  <p class="farm-voice__status" data-status role="status" aria-live="polite"></p>
  <p class="farm-voice__privacy" data-privacy></p>
</section>
""",
    css="""
.farm-voice {
  color: var(--st-text-color, #24352b);
  font: inherit;
}
.farm-voice__controls {
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem;
}
.farm-voice__button {
  border: 1px solid #b8c9bd;
  border-radius: 0.65rem;
  background: #fff;
  color: inherit;
  cursor: pointer;
  font: inherit;
  min-height: 2.6rem;
  padding: 0.55rem 0.9rem;
}
.farm-voice__button--primary {
  background: #1f6d45;
  border-color: #1f6d45;
  color: #fff;
}
.farm-voice__button:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}
.farm-voice__status {
  margin: 0.65rem 0 0;
  min-height: 1.5rem;
}
.farm-voice__privacy {
  color: #5d6d62;
  font-size: 0.88rem;
  margin: 0.25rem 0 0;
}
""",
    js="""
export default function (component) {
  const { data, parentElement, setTriggerValue } = component
  const root = parentElement.querySelector(".farm-voice")
  if (!root) return

  const incomingData = data || {}
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition
  root.__voiceData = {
    ...incomingData,
    recognitionSupported: !!Recognition && !!window.isSecureContext,
    synthesisSupported: !!window.speechSynthesis && !!window.SpeechSynthesisUtterance,
  }
  const currentData = root.__voiceData
  const startButton = root.querySelector('[data-action="start"]')
  const stopButton = root.querySelector('[data-action="stop"]')
  const speakButton = root.querySelector('[data-action="speak"]')
  const status = root.querySelector("[data-status]")
  const privacy = root.querySelector("[data-privacy]")
  const labels = currentData.labels || {}

  startButton.textContent = labels.start || "Start microphone"
  stopButton.textContent = labels.stop || "Stop"
  speakButton.textContent = labels.speak || "Speak answer"
  privacy.textContent = labels.privacy || ""
  startButton.disabled = !currentData.recognitionSupported
  speakButton.disabled = !currentData.synthesisSupported
  stopButton.disabled = !root.__voiceState?.listening && !root.__voiceState?.speaking

  const unavailable = []
  if (!currentData.recognitionSupported && labels.recognitionUnsupported) {
    unavailable.push(labels.recognitionUnsupported)
  }
  if (!currentData.synthesisSupported && labels.synthesisUnsupported) {
    unavailable.push(labels.synthesisUnsupported)
  }
  if (unavailable.length) {
    status.textContent = unavailable.join(" ")
  } else if (!status.textContent) {
    status.textContent = labels.ready || ""
  }

  if (startButton.dataset.bound === "true") return
  startButton.dataset.bound = "true"

  root.__voiceState = { recognition: null, listening: false, speaking: false }

  startButton.addEventListener("click", () => {
    const currentData = root.__voiceData
    const currentLabels = currentData.labels || {}
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SpeechRecognition || !window.isSecureContext) {
      status.textContent = currentLabels.recognitionUnsupported || ""
      return
    }
    if (root.__voiceState.recognition) {
      root.__voiceState.recognition.abort()
    }

    const recognition = new SpeechRecognition()
    root.__voiceState.recognition = recognition
    recognition.lang = currentData.language || "en-IN"
    recognition.continuous = false
    recognition.interimResults = false
    recognition.maxAlternatives = 1
    recognition.onstart = () => {
      root.__voiceState.listening = true
      startButton.disabled = true
      stopButton.disabled = false
      status.textContent = currentLabels.listening || ""
    }
    recognition.onresult = event => {
      const transcript = event.results?.[0]?.[0]?.transcript?.trim() || ""
      if (transcript) {
        setTriggerValue("transcript", transcript)
        status.textContent = currentLabels.transcriptReceived || ""
      }
    }
    recognition.onerror = event => {
      const errorLabels = {
        "not-allowed": currentLabels.permissionDenied,
        "service-not-allowed": currentLabels.permissionDenied,
        "audio-capture": currentLabels.noMicrophone,
        "no-speech": currentLabels.noSpeech,
        "network": currentLabels.networkError,
        "aborted": currentLabels.stopped,
      }
      status.textContent = errorLabels[event.error] || currentLabels.recognitionError || ""
    }
    recognition.onend = () => {
      root.__voiceState.listening = false
      startButton.disabled = !currentData.recognitionSupported
      stopButton.disabled = !root.__voiceState.speaking
    }
    try {
      recognition.start()
    } catch (error) {
      root.__voiceState.listening = false
      startButton.disabled = false
      stopButton.disabled = true
      status.textContent = currentLabels.recognitionError || ""
    }
  })

  stopButton.addEventListener("click", () => {
    const currentLabels = root.__voiceData.labels || {}
    if (root.__voiceState.recognition && root.__voiceState.listening) {
      root.__voiceState.recognition.stop()
    }
    if (window.speechSynthesis) window.speechSynthesis.cancel()
    root.__voiceState.speaking = false
    stopButton.disabled = true
    status.textContent = currentLabels.stopped || ""
  })

  speakButton.addEventListener("click", () => {
    const currentData = root.__voiceData
    const currentLabels = currentData.labels || {}
    if (!window.speechSynthesis || !window.SpeechSynthesisUtterance) {
      status.textContent = currentLabels.synthesisUnsupported || ""
      return
    }
    const text = (currentData.speechText || "").trim()
    if (!text) {
      status.textContent = currentLabels.noAnswer || ""
      return
    }
    window.speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.lang = currentData.language || "en-IN"
    utterance.onstart = () => {
      root.__voiceState.speaking = true
      stopButton.disabled = false
      status.textContent = currentLabels.speaking || ""
    }
    utterance.onend = () => {
      root.__voiceState.speaking = false
      stopButton.disabled = root.__voiceState.listening
      status.textContent = currentLabels.speechComplete || ""
    }
    utterance.onerror = () => {
      root.__voiceState.speaking = false
      stopButton.disabled = root.__voiceState.listening
      status.textContent = currentLabels.synthesisError || ""
    }
    try {
      window.speechSynthesis.speak(utterance)
    } catch (error) {
      status.textContent = currentLabels.synthesisError || ""
    }
  })
}
""",
)


def browser_voice_controls(
    *,
    language: str,
    speech_text: str,
    labels: dict[str, str],
    key: str = "farm_ai_voice_controls",
):
    """Render controls and return the transcript from the latest recognition event."""
    result = _VOICE_WIDGET(
        key=key,
        data={
            "language": language,
            "speechText": speech_text,
            "labels": labels,
        },
        on_transcript_change=lambda: None,
    )
    return result
