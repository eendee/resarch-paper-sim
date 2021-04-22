import { Component, OnInit } from '@angular/core';
import { Paper, ParagraphString } from 'src/app/common/paper';
import { ApiResquestService } from 'src/app/services/api-request-service';
import { MatOptionSelectionChange } from '@angular/material/core';
import { Result, ResultSet } from 'src/app/common/results';

@Component({
  selector: 'app-home-page',
  templateUrl: './home-page.component.html',
  styleUrls: ['./home-page.component.css']
})
export class HomePageComponent implements OnInit {

  papers: Paper[];
  sourcePaper:Paper;
  sourceParagraphs: ParagraphString[];

  comparisonResults : Result[][];
  paragraphComparisonResults : Result[];

  targetPaper:Paper;
  targetParagraphs: ParagraphString[];
  targetParagraphsCopy: ParagraphString[];

  hideSpinner = true;
  hideSourceSpinner = true;
  hideTargetSpinner = true;

  isInCompareMode = false;

  comparedParagraph: number = 0;

  constructor(private apiService: ApiResquestService) { }

  ngOnInit(): void {
    this.loadPapers();
  }

  onPaperSelectedChanged(event: MatOptionSelectionChange, input: number): MatOptionSelectionChange {
    if (event.isUserInput) {      
      this.isInCompareMode = false;
      if(input == 1){
        this.hideSourceSpinner = false;
        this.sourcePaper = event.source.value;
        console.log(this.sourcePaper);
      }
      else{
        this.hideTargetSpinner = false;
        this.targetPaper = event.source.value;
      }
      this.apiService.getReseachPaperById(event.source.value.id).subscribe(
        (data)=>{
          if(input == 1){
            this.sourceParagraphs = data.dataset;
            this.setParagraphIds(this.sourceParagraphs);
            this.hideSourceSpinner = true;
          }
          else{
            this.targetParagraphs =  data.dataset;
            this.setParagraphIds(this.targetParagraphs);

            this.targetParagraphsCopy = data.dataset;
            this.setParagraphIds(this.targetParagraphsCopy);
            this.hideTargetSpinner = true;
          }
        },
        (error)=>{
          console.log(error)
        }
      )
    }

    return event;
  }

  setParagraphIds(paragraphs: ParagraphString[]){
    let counter = 0;
    paragraphs.forEach((x)=>{
        x.Id = counter;
        counter ++
    })
  }

  loadPapers(){
    this.apiService.getResearchPapersList().subscribe(
      (data)=>{
        this.papers = data.dataset;
      },
      (error)=>{
        console.log(error)
      }
    )
  }

  comparePapers(){
    this.hideSpinner = false;
    this.apiService.getResults(this.sourcePaper.id, this.targetPaper.id).subscribe(
      (data)=>{
        console.log(data)
        this.comparisonResults = data.results;
        console.log(this.comparisonResults)
        this.hideSpinner = true;        
        this.isInCompareMode = true;
        this.showResults();
      },
      (error)=>{
        console.log(error)
      }
    )
  }

  getParagraphResult(sourceParagraphId = 0){
    this.hideSpinner = false;
    this.apiService.getResultsPerParagraph(this.sourcePaper.id, this.targetPaper.id, sourceParagraphId.toString() ).subscribe(
      (data)=>{
        this.paragraphComparisonResults = data.results;
        
        console.log(data);
        console.log(this.paragraphComparisonResults);
        console.log(this.targetParagraphsCopy);

        this.hideSpinner = true;        
        this.isInCompareMode = true;
        let counter = 0;
        this.targetParagraphsCopy =  JSON.parse(JSON.stringify(this.targetParagraphs));
        this.targetParagraphsCopy.forEach((x)=>{
            x.score = parseFloat( this.paragraphComparisonResults[counter].avg_max_score ) * 100;
            x.originalParagraphId = counter;
            counter ++;
        })
        this.targetParagraphsCopy.sort((x,y)=>y.score - x.score)
      },
      (error)=>{
        console.log(error)
      }
    )
  }

  showResults(sourceParagraphId = 0){
    let counter = 0;
    this.targetParagraphsCopy =  JSON.parse(JSON.stringify(this.targetParagraphs));
    this.targetParagraphsCopy.forEach((x)=>{
        x.score = parseFloat( this.comparisonResults[sourceParagraphId][counter].avg_max_score ) * 100;
        x.originalParagraphId = counter;
        counter ++;
    })
    this.targetParagraphsCopy.sort((x,y)=>y.score - x.score)

  }

  compareParagraph(paragraphId: number){
    this.sourceParagraphs.forEach(x=>x.isInSpotlight = false);
    this.sourceParagraphs[paragraphId].isInSpotlight = true;
    this.showResults(paragraphId)
  }

  // loadSinglePaper(paperId: string) : ParagraphString[]{
  //   this.apiService.getReseachPaperById(paperId).subscribe(
  //     (data)=>{
  //       console.log(data);
  //       return data.dataset;
  //     },
  //     (error)=>{
  //       console.log(error)
  //     }
  //   )
  //   return null;
  // }

}
