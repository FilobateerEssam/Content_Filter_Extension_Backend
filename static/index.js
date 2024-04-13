
const MOBILENET_MODEL_PATH='http://127.0.0.1:5500/mobile_tfjs/model.json'
let mobilenet;
let image;



(async function(){
  console.log('Loading model...');
  document.getElementById('loading').style.display="block";
  mobilenet = await tf.loadGraphModel(MOBILENET_MODEL_PATH);
  document.getElementById('loading').style.display="none";
  console.log('Loaded');
})();



document.getElementById('upload').addEventListener('change', function() {
    const file = this.files[0];
    image=new Image();
    
    if (file) {
      const reader = new FileReader();
      reader.onload = function() {
        const imagePreview = document.getElementById('imagePreview');
        imagePreview.innerHTML = `<img src="${reader.result}" class="img-fluid" style="max-width: 100px;">`;
        image.src=reader.result;
        // Simulating processing of the loaded image
        const resultText = document.getElementById('resultText');
        resultText.innerHTML = `<p>Image processed successfully!</p>`;
      };
      reader.readAsDataURL(file);
    }
  });

  document.getElementById('predict').onclick= async function(){
    let tensor=tf.browser.fromPixels(image).resizeNearestNeighbor([224,224])
    .toFloat()
    .expandDims();
    let prediction=await mobilenet.predict(tensor).data();
    console.log(prediction);
    
    //document.getElementById("resultText").innerHTML=prediction;

   

    // Now `numbersArray` contains the two numbers as separate elements
    const nonViolence =prediction[0];
    const violence = prediction[1];
    r= document.getElementById("resultText");
    if(nonViolence>violence){
     r.innerHTML="Non violence";
    }
    else{
      r.innerHTML="violence";
    }
  }

  