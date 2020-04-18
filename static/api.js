function queryDatabase(string, callack) {
      $.ajax({
        url: "/query/",
        data: JSON.stringify({
          query: string
          }),
        dataType: 'json',
        contentType: 'application/json;charset=UTF-8',
        type: 'POST',
        success: function (e) {
          callack(e);
        },
        error: function (jqXHR, textStatus, errorThrown ) {
          console.log("Error", jqXHR, textStatus, errorThrown);
        },
      });
    }

function createImageCard(location, thumbnail, cid){
    let car_tmpl = `<div class="card" style="width: 10rem;">
      <img class="card-img-top" src="`+thumbnail+`" alt="Card image cap">
      <div class="card-body">
        <p class="card-text"> Movie:`+location.movie+` FramePos:`+location.frame_pos+`</p>
        <a id="submit-` + cid + `" class="btn btn-primary">Submit</a>
      </div>
    </div>`;
    return car_tmpl;
}


function submitResult(movie, frame_pos){
  console.log( "/submit/" + movie + "/" + frame_pos + "/")
      $.ajax({
        url: "/submit/" + movie + "/" + frame_pos + "/",
        dataType: 'json',
        contentType: 'application/json;charset=UTF-8',
        type: 'GET',
        error: function (jqXHR, textStatus, errorThrown ) {
          console.log("Error", jqXHR, textStatus, errorThrown);
        },
      });
}