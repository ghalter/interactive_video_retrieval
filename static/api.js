function queryDatabase(string, sub, callack) {
      $.ajax({
        url: "/query/",
        data: JSON.stringify({
          subquery:sub,
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
    <a id="submit-` + cid + `" class="btn btn-primary">
      <img class="card-img-top" src="`+thumbnail+`" alt="Card image cap">
</a>
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

function sendImage(canvas, sub, callback){
  var dataURL = canvas.toDataURL();
  $.ajax({
      type: "POST",
      url: "/query-image/",
      data: {
        subquery:JSON.stringify(sub),
        imageBase64: dataURL
      },
      success: function (e) {
        callback(e);
      },
      error: function (jqXHR, textStatus, errorThrown ) {
        console.log("Error", jqXHR, textStatus, errorThrown);
      },
    }).done(function() {
      console.log('sent');
  });
}