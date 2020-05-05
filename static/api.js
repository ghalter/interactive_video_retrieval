function queryDatabase(string, sub, callack) {
  $.ajax({
    url: "/query/",
    data: JSON.stringify({
      subquery: sub,
      query: string
    }),
    dataType: 'json',
    contentType: 'application/json;charset=UTF-8',
    type: 'POST',
    success: function (e) {
      callack(e);
    },
    error: function (jqXHR, textStatus, errorThrown) {
      console.log("Error", jqXHR, textStatus, errorThrown);
    },
  });
}

function findSimilar(elem, callack) {
  $.ajax({
    url: "/similar/",
    data: JSON.stringify({
      query: elem
    }),
    dataType: 'json',
    contentType: 'application/json;charset=UTF-8',
    type: 'POST',
    success: function (e) {
      callack(e);
    },
    error: function (jqXHR, textStatus, errorThrown) {
      console.log("Error", jqXHR, textStatus, errorThrown);
    },
  });
}

function createImageCard(type, location, thumbnail, cid) {
  let car_tmpl = `<div class="card wrapper" style="width: 10rem;">
    <a id="`+ type + `-` + cid + `" class="btn btn-primary">
      <img class="card-img-top" src="`+ thumbnail + `" alt="Card image cap"></a>
      <a  id="`+ type + `-` + cid + `-similar" class="button btn-warning text-center btn-md" ><h5>Find Similar</h5></a>
      </div>
    </div>`;
  return car_tmpl;
}

// function createImageCard(type, location, thumbnail, cid){
//   let car_tmpl = `<div class="card" style="width: 10rem;">
//   <a id="`+type+`-` + cid + `" class="btn btn-primary">
//     <img class="card-img-top" src="`+thumbnail+`" alt="Card image cap">
// </a>
//     </div>
//     <div>
//   </div>`;
//   return car_tmpl;
// }



function submitResult(movie, frame_pos) {
  console.log("/submit/" + movie + "/" + frame_pos + "/")
  $.ajax({
    url: "/submit/" + movie + "/" + frame_pos + "/",
    dataType: 'json',
    contentType: 'application/json;charset=UTF-8',
    type: 'GET',
    error: function (jqXHR, textStatus, errorThrown) {
      console.log("Error", jqXHR, textStatus, errorThrown);
    },
  });
}

function sendImage(canvas, sub, callback) {
  var dataURL = canvas.toDataURL();
  $.ajax({
    type: "POST",
    url: "/query-image/",
    data: {
      subquery: JSON.stringify(sub),
      imageBase64: dataURL
    },
    success: function (e) {
      callback(e);
    },
    error: function (jqXHR, textStatus, errorThrown) {
      console.log("Error", jqXHR, textStatus, errorThrown);
    },
  }).done(function () {
    console.log('sent');
  });
}