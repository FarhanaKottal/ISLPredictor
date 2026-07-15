const video = document.getElementById("video");
const canvas = document.getElementById("canvas");

const predictionText = document.getElementById("prediction");
const confidenceText = document.getElementById("confidence");

navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => {
    video.srcObject = stream;
});

const validGestures = [
    "Hello",
    "IloveYou",
    "No",
    "Please",
    "Thanks",
    "Yes"
];

let lastSpoken = "";

function speak(text) {

    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);

    utterance.lang = "en-US";

    window.speechSynthesis.speak(utterance);

}

async function captureFrame() {

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video,0,0);

    canvas.toBlob(async(blob)=>{

        const formData = new FormData();

        formData.append("image",blob,"frame.jpg");

        const response = await fetch("/predict",{
            method:"POST",
            body:formData
        });

        const data = await response.json();

        predictionText.innerHTML =
            data.gesture === ""
            ? "Prediction : --"
            : "Prediction : " + data.gesture;

        confidenceText.innerHTML =
            data.gesture === ""
            ? ""
            : "Confidence : " + data.confidence + "%";

        // Speak ONLY valid gestures
        if(validGestures.includes(data.gesture)){

            if(lastSpoken !== data.gesture){

                speak(data.gesture);

                lastSpoken = data.gesture;

            }

        }else{

            // Reset only when no gesture
            lastSpoken = "";

        }

    },"image/jpeg");

}

video.onloadedmetadata = ()=>{

    setInterval(captureFrame,1000);

};