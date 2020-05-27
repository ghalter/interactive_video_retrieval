var currentSubmition = []
getSubmitionsFromServer();

function updateSubmitionView(){
    $("#submition-list").html("");
    let cid = 0;
    currentSubmition.forEach((elem)=>{
        let t = createImageCard("submit", elem.location, elem.thumbnail, cid);
        $("#submition-list").append(t);
        
        $("#submit-" + cid +"-similar").on("click", function(){
            findSimilar(elem, queryCallback);
        });

        $("#submit-" + cid +"-movie").on("click", function(){
            findMovie(elem, queryCallback);
        });

        $("#submit-" + cid).on("click", function(){
            submitResult(elem.location.movie, elem.location.frame_pos)
        });
        cid ++;
    });
}

function getSubmitionsFromServer(){
    $.ajax({
        url: "/get-bookmarks/",
        dataType: 'json',
        contentType: 'application/json;charset=UTF-8',
        type: 'GET',
        success: function(answer){
            if (JSON.stringify(answer) != JSON.stringify(currentSubmition)){
                currentSubmition = answer
                updateSubmitionView();
            }

        },
        error: function (jqXHR, textStatus, errorThrown) {
          console.log("Error", jqXHR, textStatus, errorThrown);
        },
        complete: function(){
            setTimeout(function(){
                getSubmitionsFromServer();
            }, 500)
        }
      });
}

function pushSubmitionsToServer(){
    let entries = []
    currentSubmition.forEach((elem) => {
        entries.push(elem.id)
    })
    $.ajax({
        url: "/update-bookmarks/",
        data: JSON.stringify({
          bookmarks: entries,
        }),
        dataType: 'json',
        contentType: 'application/json;charset=UTF-8',
        type: 'POST',
        success: function (e) {
        },
        error: function (jqXHR, textStatus, errorThrown) {
          console.log("Error", jqXHR, textStatus, errorThrown);
        },
      });
}


function clearSubmition(){
    currentSubmition = []
    updateSubmitionView();
    pushSubmitionsToServer();
}

function addToSubmitionList(t){
    currentSubmition.push(t)
    updateSubmitionView();
    pushSubmitionsToServer();
}
