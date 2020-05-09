var currentResult = []
var previousHistory = []

function pushHistory(type="Query"){
    t = []
    let time = new Date(Date.now()).toTimeString()
    currentResult.forEach((elem) =>{
        t.push(elem)
    })
    previousHistory.push([type, time, t])
    updateHistoryView();
}

function updateHistoryView(){
    $("#history-list").html("")
    let idx = 0;
    previousHistory.forEach((entry) => {
        console.log(entry)
        let time = entry[1]
        let type = entry[0]
        let elem = entry[2]

    
        var name = 'btn-history-'+ idx;
        var html =  '<li class="list-group-item list-group-flush"><button class="btn btn-link collapsed" id="'+name + '">'+type + time +': </button></li>'
        $("#history-list").prepend(html)
        let nd = idx
        $("#"+name).on("click", function(){
            loadHistoryPoint(nd)
        })
        idx ++;
    })
}

function loadHistoryPoint(idx){
    currentResult = previousHistory[idx][2]
    updateResultView();
}
