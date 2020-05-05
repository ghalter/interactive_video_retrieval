var currentResult = []
var previousHistory = []

function pushHistory(type="Query"){
    t = []
    currentResult.forEach((elem) =>{
        t.push(elem)
    })
    previousHistory.push(t)
    let now = new Date();
    var idx = previousHistory.length - 1;
    var name = 'btn-history-'+ idx;
    var html =  '<li class="list-group-item list-group-flush"><button class="btn btn-link collapsed" id="'+name + '">'+type+": " +now+' </button></li>'
    $("#history-list").prepend(html)

    $("#"+name).on("click", function(){
        loadHistoryPoint(idx)
    })
}

function loadHistoryPoint(idx){
    currentResult = previousHistory[idx]
    updateResultView();
}
