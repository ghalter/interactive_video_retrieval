var currentSubmition = []

function updateSubmitionView(){
    $("#submition-list").html("");
    let cid = 0;
    console.log(currentSubmition)
    currentSubmition.forEach((elem)=>{
        let t = createImageCard("submit", elem.location, elem.thumbnail, cid);
        $("#submition-list").append(t);
        
        $("#submit-" + cid +"-similar").on("click", function(){
            findSimilar(elem, queryCallback);
        });
        $("#submit-" + cid).on("click", function(){
            submitResult(elem.location.movie, elem.location.frame_pos)
        });
        cid ++;
        console.log("Done")
    });
}

function getSubmitionsFromServer(){

}

function pushSubmitionsToServer(){

}


function clearSubmition(){
    currentSubmition = []
    updateSubmitionView();
}

function addToSubmitionList(t){
    currentSubmition.push(t)
    updateSubmitionView();
}
