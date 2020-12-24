
// copy text on tagging algorithm page
function myFunction() {
    var copyText = document.getElementById("exampleFormControlTextarea1");
    copyText.select();
    copyText.setSelectionRange(0, 99999)
    document.execCommand("copy");
    alert("Copied the text: " + copyText.value);
  }

  // loading spinner
  function hideSpinner() { 
    document.getElementById('spinner') 
            .style.display = 'none'; 
}